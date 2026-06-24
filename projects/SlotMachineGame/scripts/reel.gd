class_name Reel
extends Node2D

# Tek reel: bir Sprite2D gösterir, spin_to ile animasyonlu döner, durunca stopped sinyali yayar.

signal stopped(symbol)

var is_spinning: bool = false

@onready var display: Sprite2D = $Display

var _rng := RandomNumberGenerator.new()
var _symbols: SymbolSet


func _ready() -> void:
	_rng.randomize()


func set_symbols(sym_set: SymbolSet) -> void:
	_symbols = sym_set


func set_symbol(sym) -> void:
	display.texture = sym.texture
	display.modulate = Color.WHITE


# Hedef sembole duration süresinde dön. Önce hızlı rastgele cycle, sonra ease-out stop.
func spin_to(target, duration: float) -> void:
	if is_spinning:
		return
	is_spinning = true
	display.modulate = Color(1, 1, 1, 0.6)
	_cycle_animations(duration, target)


func _cycle_animations(duration: float, target) -> void:
	var elapsed: float = 0.0
	var interval: float = 0.08
	while elapsed < duration - 0.28:
		display.texture = _symbols.get_random_any().texture
		await get_tree().create_timer(interval).timeout
		elapsed += interval
	# Son iki kare: ease-out hissi — yavaşla
	display.texture = _symbols.get_random_any().texture
	await get_tree().create_timer(0.12).timeout
	display.texture = _symbols.get_random_any().texture
	await get_tree().create_timer(0.16).timeout
	# Dur
	display.texture = target.texture
	display.modulate = Color.WHITE
	is_spinning = false
	stopped.emit(target)
