extends Node2D

# LUCKY SLOTS — MVP
# Wild, Scatter/Free Spins, Save, Prosedürel ses.

var _symbols: SymbolSet

@onready var reel1: Reel = $Reels/Reel1
@onready var reel2: Reel = $Reels/Reel2
@onready var reel3: Reel = $Reels/Reel3
@onready var spin_button: Button = $UI/SpinButton
@onready var balance_label: Label = $UI/BalanceLabel
@onready var result_label: Label = $UI/ResultLabel
@onready var bet_slider: HSlider = $UI/BetSlider
@onready var bet_label: Label = $UI/BetLabel
@onready var free_spins_label: Label = $UI/FreeSpinsLabel
@onready var mute_button: Button = $UI/MuteButton
@onready var daily_bonus_panel: Panel = $DailyBonusPanel
@onready var claim_bonus_button: Button = $DailyBonusPanel/ClaimBonusButton

var current_bet: int = 10
var is_spinning: bool = false
var in_free_spins: bool = false


func _ready() -> void:
	_symbols = SymbolSet.new()
	for r in [reel1, reel2, reel3]:
		r.set_symbols(_symbols)
	# İlk semboller
	reel1.set_symbol(_symbols.get_symbol("seven"))
	reel2.set_symbol(_symbols.get_symbol("cherry"))
	reel3.set_symbol(_symbols.get_symbol("lemon"))
	# UI
	spin_button.pressed.connect(_on_spin_pressed)
	bet_slider.value_changed.connect(_on_bet_changed)
	mute_button.pressed.connect(_on_mute_pressed)
	claim_bonus_button.pressed.connect(_on_claim_bonus)
	# Save load
	SaveManager.load_game()
	# Müzik başlat
	AudioManager.start_music()
	_update_ui()
	_check_daily_bonus()


func _on_spin_pressed() -> void:
	if is_spinning:
		return
	AudioManager.play_click()
	# Free spin kontrolü
	var using_free_spin := in_free_spins and SaveManager.free_spins_remaining > 0
	if not using_free_spin:
		if SaveManager.balance < current_bet:
			result_label.text = "Insufficient Balance!"
			_check_daily_bonus()
			return
		SaveManager.balance -= current_bet
	else:
		SaveManager.free_spins_remaining -= 1

	is_spinning = true
	spin_button.disabled = true
	result_label.text = "Spinning..."
	AudioManager.play_spin()
	SaveManager.total_spins += 1
	_update_ui()

	# Hedef semboller
	var s1 := _symbols.get_random_any()
	var s2 := _symbols.get_random_any()
	var s3 := _symbols.get_random_any()

	# Paralel spin (kademeli duration)
	reel1.spin_to(s1, 1.0)
	reel2.spin_to(s2, 1.25)
	reel3.spin_to(s3, 1.5)

	# Reel stop sesleri
	await get_tree().create_timer(1.0).timeout
	AudioManager.play_reel_stop()
	await get_tree().create_timer(0.25).timeout
	AudioManager.play_reel_stop()
	await get_tree().create_timer(0.25).timeout
	AudioManager.play_reel_stop()

	# Tüm reel'lerin durmasını bekle (en uzun 1.5s)
	await get_tree().create_timer(1.5).timeout

	# Kazanç hesabı
	var win_amount := _calculate_win(s1, s2, s3)
	var scatter_count := _count_scatter(s1, s2, s3)

	# Scatter tetiklenmesi (3 scatter)
	if scatter_count >= 3:
		if in_free_spins:
			SaveManager.free_spins_remaining += 5
			result_label.text = "5 EXTRA FREE SPINS!"
		else:
			in_free_spins = true
			SaveManager.free_spins_remaining = 10
			result_label.text = "10 FREE SPINS!"
		AudioManager.play_free_spins_trigger()
	elif win_amount > 0:
		SaveManager.balance += win_amount
		if win_amount > SaveManager.biggest_win:
			SaveManager.biggest_win = win_amount
		result_label.text = "WIN! +%d" % win_amount
		if win_amount >= current_bet * 20:
			AudioManager.play_win_big()
			_flash_win()
		else:
			AudioManager.play_win_small()
	else:
		result_label.text = "Try Again!"
		AudioManager.play_lose()

	# Free spin modu bitişi
	if in_free_spins and SaveManager.free_spins_remaining <= 0:
		in_free_spins = false
		result_label.text = "Free Spins End!"

	is_spinning = false
	spin_button.disabled = false
	SaveManager.save_game()
	_update_ui()
	_check_daily_bonus()


func _calculate_win(s1, s2, s3) -> int:
	# Wild substitute: wild, herhangi normal sembolün yerine geçer (scatter hariç)
	# 3-match: üçü de aynı (wild'lar fill eder)
	# 2-match: iki sembol aynı (wild fill), üçüncü farklı

	# Scatter'a göre kazanç olmaz (scatter free spin tetikler)
	if s1.is_scatter and s2.is_scatter and s3.is_scatter:
		return 0

	# Wild substitute ile 3-match kontrolü
	var payout := _three_match_payout(s1, s2, s3)
	if payout > 0:
		return current_bet * payout

	# 2-match (wild substitute dahil)
	payout = _two_match_payout(s1, s2, s3)
	if payout > 0:
		return int(round(current_bet * payout * 0.2))

	return 0


# Üç sembolün wild substitute ile 3-match olup olmadığını kontrol et.
# Wild 3-match'i (üçü de wild) en yüksek payout (50) say.
func _three_match_payout(s1, s2, s3) -> int:
	# Tümü scatter ise 0 (free spin ayrı işlenir)
	if s1.is_scatter and s2.is_scatter and s3.is_scatter:
		return 0
	# Scatter varsa normal 3-match bozulur (scatter substitute olmaz)
	if s1.is_scatter or s2.is_scatter or s3.is_scatter:
		return 0

	var normals := []
	for s in [s1, s2, s3]:
		if not s.is_wild:
			normals.append(s)

	if normals.size() == 3:
		# Wild yok — üçü aynı mı?
		if s1.name == s2.name and s2.name == s3.name:
			return s1.payout
	elif normals.size() == 2:
		# Bir wild, iki normal — iki normal aynı mı?
		if normals[0].name == normals[1].name:
			return normals[0].payout
	elif normals.size() == 1:
		# İki wild, bir normal
		return normals[0].payout
	elif normals.size() == 0:
		# Üçü de wild
		return 50  # Wild 3-match = en yüksek
	return 0


# Wild substitute ile 2-match: en az iki sembol aynı (wild fill eder), üçüncü farklı.
func _two_match_payout(s1, s2, s3) -> int:
	# Her çifti kontrol et, wild substitute uygula
	var pairs := [[s1, s2], [s2, s3], [s1, s3]]
	for pair in pairs:
		var a = pair[0]
		var b = pair[1]
		# Scatter'lar 2-match'e girmez
		if a.is_scatter or b.is_scatter:
			continue
		# İkisi de aynı normal sembol
		if not a.is_wild and not b.is_wild:
			if a.name == b.name:
				return a.payout
		# Biri wild, diğeri normal
		elif a.is_wild and not b.is_wild:
			return b.payout
		elif b.is_wild and not a.is_wild:
			return a.payout
		# İkisi de wild — 2-match sayılmaz (üçüncüyle 3-match zaten kontrol edildi)
	return 0


func _count_scatter(s1, s2, s3) -> int:
	var c := 0
	for s in [s1, s2, s3]:
		if s.is_scatter:
			c += 1
	return c


func _flash_win() -> void:
	for i in range(3):
		result_label.modulate = Color(1, 0.84, 0)
		await get_tree().create_timer(0.15).timeout
		result_label.modulate = Color.WHITE
		await get_tree().create_timer(0.15).timeout


func _on_bet_changed(value: float) -> void:
	current_bet = int(value)
	AudioManager.play_click()
	_update_ui()


func _on_mute_pressed() -> void:
	var muted := AudioManager.toggle_mute()
	mute_button.text = "S" if not muted else "X"


func _on_claim_bonus() -> void:
	SaveManager.reset_with_bonus(500)
	daily_bonus_panel.visible = false
	AudioManager.play_win_small()
	_update_ui()


func _check_daily_bonus() -> void:
	if SaveManager.balance <= 0 and not in_free_spins:
		daily_bonus_panel.visible = true


func _update_ui() -> void:
	balance_label.text = "Balance: %d" % SaveManager.balance
	bet_label.text = "Bet: %d" % current_bet
	free_spins_label.visible = in_free_spins or SaveManager.free_spins_remaining > 0
	free_spins_label.text = "FREE SPINS: %d" % SaveManager.free_spins_remaining
	# Free spin modunda slider disabled
	bet_slider.editable = not in_free_spins
	# Max bet balance'e göre
	if not in_free_spins:
		bet_slider.max_value = min(100, max(1, SaveManager.balance))
		if current_bet > SaveManager.balance and SaveManager.balance > 0:
			current_bet = SaveManager.balance
			bet_slider.value = current_bet
