class_name SymbolSet
extends RefCounted

# Sembol veri sınıfı + registry. Wild/scatter dahil tüm sembolleri yükler.

class SymbolData:
	var name: String
	var texture: Texture2D
	var payout: int
	var is_wild: bool
	var is_scatter: bool

	func _init(p_name: String, p_texture: Texture2D, p_payout: int, p_wild: bool, p_scatter: bool) -> void:
		name = p_name
		texture = p_texture
		payout = p_payout
		is_wild = p_wild
		is_scatter = p_scatter

# (name, payout, is_wild, is_scatter) — SVG yolu "res://assets/symbols/<name>.svg"
const _DEFS := [
	["seven", 50, false, false],
	["diamond", 30, false, false],
	["bell", 20, false, false],
	["bar", 15, false, false],
	["grape", 10, false, false],
	["orange", 8, false, false],
	["lemon", 5, false, false],
	["cherry", 3, false, false],
	["wild", 50, true, false],
	["scatter", 0, false, true],
]

# Normal sembol ağırlıkları (wild/scatter hariç). Yüksek payout = düşük frekans.
const _NORMAL_WEIGHTS := {
	"seven": 1,
	"diamond": 2,
	"bell": 3,
	"bar": 4,
	"grape": 5,
	"orange": 6,
	"lemon": 7,
	"cherry": 8,
}

# Wild/scatter nadir görünsün.
const _WILD_WEIGHT := 2
const _SCATTER_WEIGHT := 3

var _by_name: Dictionary = {}
var _normal_pool: Array = []
var _full_pool: Array = []
var _rng := RandomNumberGenerator.new()


func _init() -> void:
	_rng.randomize()
	for d in _DEFS:
		var name: String = d[0]
		var payout: int = d[1]
		var is_wild: bool = d[2]
		var is_scatter: bool = d[3]
		var tex := load("res://assets/symbols/%s.svg" % name) as Texture2D
		var sym := SymbolData.new(name, tex, payout, is_wild, is_scatter)
		_by_name[name] = sym
		if not is_wild and not is_scatter:
			for _i in range(_NORMAL_WEIGHTS.get(name, 1)):
				_normal_pool.append(sym)
		# Full pool: normal ağırlıklar + wild/scatter
		if is_wild:
			for _i in range(_WILD_WEIGHT):
				_full_pool.append(sym)
		elif is_scatter:
			for _i in range(_SCATTER_WEIGHT):
				_full_pool.append(sym)
		else:
			for _i in range(_NORMAL_WEIGHTS.get(name, 1)):
				_full_pool.append(sym)


func get_symbol(name: String) -> SymbolData:
	return _by_name.get(name)


func get_random_normal() -> SymbolData:
	return _normal_pool[_rng.randi_range(0, _normal_pool.size() - 1)]


func get_random_any() -> SymbolData:
	return _full_pool[_rng.randi_range(0, _full_pool.size() - 1)]
