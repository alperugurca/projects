extends Node

# Singleton (autoload). ConfigFile ile save/load.
# Yol: user://slot_save.cfg

const SAVE_PATH := "user://slot_save.cfg"
const DEFAULT_BALANCE := 1000
const DAILY_BONUS := 500

var balance: int = DEFAULT_BALANCE
var total_spins: int = 0
var biggest_win: int = 0
var free_spins_remaining: int = 0

var _config := ConfigFile.new()


func _ready() -> void:
	load_game()


func load_game() -> void:
	var err := _config.load(SAVE_PATH)
	if err == OK:
		balance = int(_config.get_value("player", "balance", DEFAULT_BALANCE))
		total_spins = int(_config.get_value("player", "total_spins", 0))
		biggest_win = int(_config.get_value("player", "biggest_win", 0))
		free_spins_remaining = int(_config.get_value("player", "free_spins_remaining", 0))
	else:
		# İlk açılış — default değerlerle kaydet
		_save_to_disk()


func save_game() -> void:
	_config.set_value("player", "balance", balance)
	_config.set_value("player", "total_spins", total_spins)
	_config.set_value("player", "biggest_win", biggest_win)
	_config.set_value("player", "free_spins_remaining", free_spins_remaining)
	_save_to_disk()


func reset_with_bonus(amount: int = DAILY_BONUS) -> void:
	balance = amount
	free_spins_remaining = 0
	save_game()


func _save_to_disk() -> void:
	_config.save(SAVE_PATH)
