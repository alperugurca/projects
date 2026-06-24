extends Node

# Singleton. AudioStreamGenerator + playback fill ile prosedürel ses.

var is_muted: bool = false

const SAMPLE_RATE := 22050

var _sfx_player := AudioStreamPlayer.new()
var _music_player := AudioStreamPlayer.new()
var _music_active: bool = false
var _music_phase: float = 0.0
var _music_freqs := [130.81, 196.00, 261.63]  # C3, G3, C4 — sakin pad

# Aktif SFX durumu (tek seferde bir SFX)
var _sfx_active: bool = false
var _sfx_samples_left: int = 0
var _sfx_total: int = 0
var _sfx_type: String = ""  # "tone","sweep","arp"
var _sfx_freq: float = 0.0
var _sfx_freq_end: float = 0.0
var _sfx_volume: float = 0.0
var _sfx_arp_freqs: Array = []
var _sfx_arp_index: int = 0
var _sfx_arp_note_samples: int = 0


func _ready() -> void:
	add_child(_sfx_player)
	add_child(_music_player)
	_music_player.volume_db = -14.0
	# SFX için generator setup
	var gen := AudioStreamGenerator.new()
	gen.mix_rate = SAMPLE_RATE
	gen.buffer_length = 0.5
	_sfx_player.stream = gen


# --- Public API ---
func play_click() -> void:
	_start_tone(1000.0, 0.03, 0.3)

func play_spin() -> void:
	_start_sweep(200.0, 600.0, 0.3, 0.3)

func play_reel_stop() -> void:
	_start_tone(800.0, 0.06, 0.25)

func play_win_small() -> void:
	_start_arpeggio([523.25, 659.25], 0.1, 0.3)

func play_win_big() -> void:
	_start_arpeggio([523.25, 659.25, 783.99, 1046.5], 0.12, 0.35)

func play_lose() -> void:
	_start_sweep(400.0, 200.0, 0.3, 0.3)

func play_free_spins_trigger() -> void:
	_start_arpeggio([523.25, 659.25, 783.99, 1046.5, 1318.5], 0.15, 0.4)

func start_music() -> void:
	if _music_active:
		return
	_music_active = true
	var gen := AudioStreamGenerator.new()
	gen.mix_rate = SAMPLE_RATE
	gen.buffer_length = 1.0
	_music_player.stream = gen
	_music_player.play()

func stop_music() -> void:
	_music_active = false
	_music_player.stop()

func toggle_mute() -> bool:
	is_muted = not is_muted
	AudioServer.set_bus_mute(0, is_muted)
	return is_muted


# --- _process: fill buffers ---
func _process(_delta: float) -> void:
	if _sfx_active:
		_fill_sfx()
	if _music_active:
		_fill_music()


func _fill_sfx() -> void:
	var playback := _sfx_player.get_stream_playback() as AudioStreamPlayback
	if playback == null:
		return
	var frames_avail: int = playback.get_frames_available()
	var frames := PackedVector2Array()
	frames.resize(frames_avail)
	for i in range(frames_avail):
		var sample := _next_sfx_sample()
		frames[i] = Vector2(sample, sample)
	playback.push_buffer(frames)


func _fill_music() -> void:
	var playback := _music_player.get_stream_playback() as AudioStreamPlayback
	if playback == null:
		return
	var frames_avail: int = playback.get_frames_available()
	var frames := PackedVector2Array()
	frames.resize(frames_avail)
	for i in range(frames_avail):
		var sample := _next_music_sample()
		frames[i] = Vector2(sample, sample)
	playback.push_buffer(frames)


func _next_sfx_sample() -> float:
	if _sfx_samples_left <= 0:
		_sfx_active = false
		return 0.0
	var t := float(_sfx_total - _sfx_samples_left) / SAMPLE_RATE
	var env := 1.0
	if _sfx_total - _sfx_samples_left < 64: env = float(_sfx_total - _sfx_samples_left) / 64.0
	elif _sfx_samples_left < 256: env = float(_sfx_samples_left) / 256.0
	var s := 0.0
	if _sfx_type == "tone":
		s = sin(t * _sfx_freq * TAU) * env * _sfx_volume
	elif _sfx_type == "sweep":
		var progress := float(_sfx_total - _sfx_samples_left) / _sfx_total
		var f := lerpf(_sfx_freq, _sfx_freq_end, progress)
		s = sin(t * f * TAU) * env * _sfx_volume
	elif _sfx_type == "arp":
		var f: float = _sfx_arp_freqs[mini(_sfx_arp_index, _sfx_arp_freqs.size() - 1)]
		s = sin(t * f * TAU) * env * _sfx_volume
	_sfx_samples_left -= 1
	# Arp note geçişi
	if _sfx_type == "arp" and _sfx_arp_note_samples > 0:
		var notes_done := _sfx_total - _sfx_samples_left
		var new_idx := notes_done / _sfx_arp_note_samples
		if new_idx != _sfx_arp_index and new_idx < _sfx_arp_freqs.size():
			_sfx_arp_index = new_idx
	return clampf(s, -1.0, 1.0)


func _next_music_sample() -> float:
	# Üç frekanslı sakin pad, düşük volume, slow LFO
	var s := 0.0
	for f in _music_freqs:
		s += sin(_music_phase * f * TAU / SAMPLE_RATE) * 0.15
	# LFO
	var lfo := sin(_music_phase * 0.2 * TAU / SAMPLE_RATE) * 0.05 + 0.95
	s *= lfo
	_music_phase += 1.0
	return clampf(s, -1.0, 1.0)


# --- SFX başlatma ---
func _start_tone(freq: float, duration: float, volume: float) -> void:
	if is_muted:
		return
	_sfx_type = "tone"
	_sfx_freq = freq
	_sfx_freq_end = freq
	_sfx_volume = volume
	_sfx_total = int(duration * SAMPLE_RATE)
	_sfx_samples_left = _sfx_total
	_sfx_arp_index = 0
	_sfx_active = true
	if not _sfx_player.playing:
		_sfx_player.play()

func _start_sweep(f_start: float, f_end: float, duration: float, volume: float) -> void:
	if is_muted:
		return
	_sfx_type = "sweep"
	_sfx_freq = f_start
	_sfx_freq_end = f_end
	_sfx_volume = volume
	_sfx_total = int(duration * SAMPLE_RATE)
	_sfx_samples_left = _sfx_total
	_sfx_active = true
	if not _sfx_player.playing:
		_sfx_player.play()

func _start_arpeggio(freqs: Array, note_duration: float, volume: float) -> void:
	if is_muted:
		return
	_sfx_type = "arp"
	_sfx_arp_freqs = freqs
	_sfx_arp_note_samples = int(note_duration * SAMPLE_RATE)
	_sfx_total = _sfx_arp_note_samples * freqs.size()
	_sfx_samples_left = _sfx_total
	_sfx_arp_index = 0
	_sfx_volume = volume
	_sfx_active = true
	if not _sfx_player.playing:
		_sfx_player.play()
