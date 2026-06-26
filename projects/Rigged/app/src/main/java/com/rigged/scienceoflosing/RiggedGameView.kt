package com.rigged.scienceoflosing

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.os.SystemClock
import android.view.MotionEvent
import android.view.View
import java.util.ArrayDeque
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.random.Random

class RiggedGameView(context: Context) : View(context) {
    private enum class GameMode { SLOTS, ROULETTE }
    private enum class BetColor { RED, BLACK }
    private enum class Diagram { BRAIN, MATH, SKINNER_BOX, ARCHITECTURE, ROULETTE }

    private data class ScienceBeat(
        val title: String,
        val line: String,
        val fact: String,
        val diagram: Diagram,
        val enlightenmentGain: Int,
    )

    private val dp = resources.displayMetrics.density
    @Suppress("DEPRECATION")
    private val sp = resources.displayMetrics.scaledDensity
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val buttons = mutableMapOf<String, RectF>()
    private val random = Random(System.currentTimeMillis())
    private val recentFacts = ArrayDeque<String>()

    private val slotSymbols = listOf("7", "BAR", "CH", "BELL", "EV", "LLN")
    private val rouletteRedNumbers = setOf(
        1, 3, 5, 7, 9, 12, 14, 16, 18,
        19, 21, 23, 25, 27, 30, 32, 34, 36,
    )

    private val scienceDeck = listOf(
        ScienceBeat(
            "Dopamine dispenser online",
            "Dispensing a micro-dose of dopamine to your ventral striatum to keep you pulling the lever. Please hold.",
            "Variable rewards activate dopamine pathways more aggressively than predictable rewards.",
            Diagram.BRAIN,
            7,
        ),
        ScienceBeat(
            "Skinner would like royalties",
            "This is a variable ratio reinforcement schedule. It is the same behavioral trap used in lab boxes, only with uglier carpet.",
            "Variable ratio schedules keep behavior persistent because the next reward always feels plausibly close.",
            Diagram.SKINNER_BOX,
            8,
        ),
        ScienceBeat(
            "Expected value has entered chat",
            "Your vibe is not a term in the expected value equation. Brutal, but tidy.",
            "EV = probability of each outcome times payout, minus the bet. Negative EV means the average player leaks money.",
            Diagram.MATH,
            7,
        ),
        ScienceBeat(
            "Architecture doing crimes",
            "No clocks, no windows, and every surface screams 'stay.' The building has a user-retention strategy.",
            "Casinos reduce temporal cues so sessions feel shorter and decisions feel less anchored.",
            Diagram.ARCHITECTURE,
            6,
        ),
        ScienceBeat(
            "C-major coping mechanism",
            "Slot sounds are tuned to make even losing rounds feel like achievement jingles. Your ears are being gaslit in C-major.",
            "Audio and light feedback can frame small losses as wins, softening the pain signal.",
            Diagram.ARCHITECTURE,
            7,
        ),
        ScienceBeat(
            "Loss aversion check",
            "The pain of losing hits harder than an equal win feels good. Naturally, your brain's solution is 'risk more.' Elegant disaster.",
            "Loss aversion makes people overreact to losses and chase recovery instead of stopping.",
            Diagram.BRAIN,
            8,
        ),
        ScienceBeat(
            "Sunk cost ceremony",
            "Money already lost is not a hostage you can rescue. It has moved on. You should try that.",
            "The sunk cost fallacy treats past losses as a reason to continue, even when future odds are still bad.",
            Diagram.MATH,
            8,
        ),
    )

    private var mode = GameMode.SLOTS
    private var selectedBet = BetColor.RED
    private var bankroll = 500
    private var enlightenment = 0
    private var totalSpins = 0
    private var lossesInRow = 0
    private var controlTaps = 0
    private var scienceIndex = 0
    private var currentSlots = listOf("7", "BAR", "EV")
    private var rouletteNumber = 0
    private var rouletteLabel = "0"
    private var rouletteColor = "Green"
    private var lastDelta = 0
    private var currentDiagram = Diagram.BRAIN
    private var narratorTitle = "Virtual Croupier initialized"
    private var narratorLine =
        "Welcome to Rigged. I am your emotionally unavailable math tutor in a waistcoat."
    private var factLine =
        "Goal: fill Enlightenment by noticing that every shiny button is stapled to a negative expected value."

    init {
        isClickable = true
        recentFacts.addLast("Slots: simplified house edge is about 12 percent per spin.")
        recentFacts.addLast("Roulette: American wheel house edge is 5.26 percent.")
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        buttons.clear()

        val w = width.toFloat()
        val h = height.toFloat()
        val margin = 18f * dp
        val headerBottom = 118f * dp
        val tabsTop = headerBottom + 8f * dp
        val tabsBottom = tabsTop + 44f * dp
        val narratorHeight = min(220f * dp, h * 0.3f)
        val narratorTop = h - narratorHeight - 18f * dp
        val gameTop = tabsBottom + 14f * dp
        val gameBottom = narratorTop - 14f * dp

        drawBackground(canvas, w, h)
        drawHeader(canvas, margin, w - margin, headerBottom)
        drawModeTabs(canvas, margin, tabsTop, w - margin, tabsBottom)

        val gameRect = RectF(margin, gameTop, w - margin, gameBottom)
        if (mode == GameMode.SLOTS) {
            drawSlots(canvas, gameRect)
        } else {
            drawRoulette(canvas, gameRect)
        }

        drawNarrator(canvas, RectF(margin, narratorTop, w - margin, h - margin))
        drawEquationOverlay(canvas, gameRect)

        postInvalidateDelayed(16L)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP) {
            val hit = buttons.entries.firstOrNull { it.value.contains(event.x, event.y) }?.key
            if (hit != null) {
                handleButton(hit)
                performClick()
                invalidate()
                return true
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun handleButton(id: String) {
        when (id) {
            "modeSlots" -> {
                mode = GameMode.SLOTS
                narratorTitle = "Table migration detected"
                narratorLine = "Changing games is charming theater. The negative EV packed a suitcase and came with you."
                factLine = "Game selection can feel like agency, but each table still defines the payout math."
                currentDiagram = Diagram.MATH
                addEnlightenment(2)
            }
            "modeRoulette" -> {
                mode = GameMode.ROULETTE
                narratorTitle = "Roulette table acquired"
                narratorLine = "A wheel is just a spreadsheet with better lighting."
                factLine = "Even-money roulette bets lose to the two green pockets over repeated trials."
                currentDiagram = Diagram.ROULETTE
                addEnlightenment(2)
            }
            "slotSpin" -> spinSlots()
            "rouletteSpin" -> spinRoulette()
            "betRed" -> chooseRouletteColor(BetColor.RED)
            "betBlack" -> chooseRouletteColor(BetColor.BLACK)
            "reset" -> resetRun()
        }
    }

    private fun spinSlots() {
        if (bankroll < SLOT_BET) {
            brokeRoast()
            return
        }

        bankroll -= SLOT_BET
        totalSpins += 1
        currentSlots = List(3) { slotSymbols[random.nextInt(slotSymbols.size)] }

        val counts = currentSlots.groupingBy { it }.eachCount()
        val isJackpot = counts["7"] == 3
        val nearJackpot = counts["7"] == 2
        val triple = counts.values.any { it == 3 }
        val pairSymbol = counts.entries.firstOrNull { it.value == 2 }?.key

        val payout = when {
            isJackpot -> 900
            triple -> 80
            pairSymbol != null && pairSymbol != "7" -> 8
            else -> 0
        }

        bankroll += payout
        lastDelta = payout - SLOT_BET

        when {
            nearJackpot -> {
                lossesInRow += 1
                teach(
                    "Near-miss hallucination",
                    "Ah, the Gambler's Fallacy. Your brain thinks you're closer to winning. Statistically, you are exactly where you started: a primate giving me your money.",
                    "Near misses can increase motivation even when each spin is independent.",
                    Diagram.BRAIN,
                    11,
                )
            }
            payout > SLOT_BET -> {
                lossesInRow = 0
                teach(
                    "Temporary loan approved",
                    "Congratulations! You have temporarily borrowed ${dollars(payout)} from the casino. The Law of Large Numbers dictates I will ask for it back.",
                    "The Law of Large Numbers pulls short-term luck toward the game's long-run average.",
                    Diagram.MATH,
                    9,
                )
            }
            payout > 0 -> {
                lossesInRow += 1
                teach(
                    "Loss disguised as confetti",
                    "Tiny payout! You still lost ${dollars(abs(lastDelta))}, but the machine clapped, so apparently that counts as closure.",
                    "Losses disguised as wins make the session feel less costly than it is.",
                    Diagram.ARCHITECTURE,
                    8,
                )
            }
            lossesInRow >= 2 -> {
                lossesInRow += 1
                teach(scienceDeck[(scienceIndex++ % scienceDeck.size)])
            }
            else -> {
                lossesInRow += 1
                teach(
                    "Dopamine dispenser online",
                    "Dispensing a micro-dose of dopamine to your ventral striatum to keep you pulling the lever. Please hold.",
                    "Slot machines use unpredictable rewards to keep attention locked on the next try.",
                    Diagram.BRAIN,
                    7,
                )
            }
        }
    }

    private fun spinRoulette() {
        if (bankroll < ROULETTE_BET) {
            brokeRoast()
            return
        }

        bankroll -= ROULETTE_BET
        totalSpins += 1

        rouletteNumber = random.nextInt(38)
        rouletteLabel = when (rouletteNumber) {
            0 -> "0"
            37 -> "00"
            else -> rouletteNumber.toString()
        }
        rouletteColor = rouletteColorFor(rouletteNumber)

        val picked = if (selectedBet == BetColor.RED) "Red" else "Black"
        val won = rouletteColor == picked
        val payout = if (won) ROULETTE_BET * 2 else 0
        bankroll += payout
        lastDelta = payout - ROULETTE_BET

        if (won) {
            lossesInRow = 0
            teach(
                "The wheel permits a snack",
                "You won ${dollars(ROULETTE_BET)}. Lovely. The expected value on this bet remains ${dollars(-105)} per ${dollars(2000)} wagered.",
                "American roulette pays even money on red or black, but 0 and 00 create a 5.26 percent house edge.",
                Diagram.ROULETTE,
                8,
            )
        } else if (rouletteColor == "Green") {
            lossesInRow += 1
            teach(
                "The green tax collects",
                "$rouletteLabel landed. The two green pockets are tiny, quiet, and extremely employed.",
                "On an American wheel, red or black wins 18 of 38 outcomes and loses 20 of 38.",
                Diagram.ROULETTE,
                10,
            )
        } else if (lossesInRow >= 2) {
            lossesInRow += 1
            teach(
                "Loss aversion check",
                "That sting you feel is loss aversion. Your brain hates losing ${dollars(ROULETTE_BET)} more than it enjoys winning the same amount.",
                "Loss aversion can push players to chase recovery after the mathematically best move is stopping.",
                Diagram.BRAIN,
                9,
            )
        } else {
            lossesInRow += 1
            teach(
                "Wheel says no",
                "The ball landed $rouletteColor. Your color loyalty has been reviewed by mathematics and found decorative.",
                "Individual roulette spins are independent; previous outcomes do not make a color due.",
                Diagram.MATH,
                7,
            )
        }
    }

    private fun chooseRouletteColor(color: BetColor) {
        if (selectedBet != color) {
            controlTaps += 1
        }
        selectedBet = color

        if (controlTaps >= 3) {
            teach(
                "Illusion of control detected",
                "Switching colors repeatedly is a ritual, not a strategy. The wheel did not receive your character development.",
                "The illusion of control makes random systems feel steerable when the outcome process has not changed.",
                Diagram.ROULETTE,
                8,
            )
            controlTaps = 0
        } else {
            narratorTitle = "Bet color selected"
            narratorLine =
                "Aesthetic update accepted. The house edge remains professionally indifferent."
            factLine = "Red and black both cover 18 pockets on a 38-pocket American wheel."
            currentDiagram = Diagram.ROULETTE
        }
    }

    private fun resetRun() {
        bankroll = 500
        enlightenment = 0
        totalSpins = 0
        lossesInRow = 0
        controlTaps = 0
        lastDelta = 0
        currentSlots = listOf("7", "BAR", "EV")
        rouletteNumber = 0
        rouletteLabel = "0"
        rouletteColor = "Green"
        narratorTitle = "Fresh bankroll, same spreadsheet"
        narratorLine =
            "Reset complete. The casino thanks you for your commitment to experimental replication."
        factLine = "A new session does not reset expected value; it only resets the user's optimism."
        currentDiagram = Diagram.MATH
        recentFacts.clear()
        recentFacts.addLast("Slots: simplified house edge is about 12 percent per spin.")
        recentFacts.addLast("Roulette: American wheel house edge is 5.26 percent.")
    }

    private fun brokeRoast() {
        teach(
            "Bankroll autopsy",
            "You are out of playable funds. That is not bad luck; that is negative expected value completing the paperwork.",
            "Bankroll depletion is the ordinary endpoint of repeated unfavorable bets.",
            Diagram.MATH,
            12,
        )
    }

    private fun teach(beat: ScienceBeat) {
        teach(beat.title, beat.line, beat.fact, beat.diagram, beat.enlightenmentGain)
    }

    private fun teach(title: String, line: String, fact: String, diagram: Diagram, gain: Int) {
        narratorTitle = title
        narratorLine = line
        factLine = fact
        currentDiagram = diagram
        addEnlightenment(gain)
        recentFacts.addFirst(fact)
        while (recentFacts.size > 3) {
            recentFacts.removeLast()
        }
    }

    private fun addEnlightenment(amount: Int) {
        val before = enlightenment
        enlightenment = (enlightenment + amount).coerceAtMost(100)
        if (before < 100 && enlightenment == 100) {
            narratorTitle = "Enlightenment achieved"
            narratorLine =
                "The neon has been stripped away. All that remains is arithmetic wearing sequins."
            factLine =
                "CBT move: label the thought, check the evidence, and let the urge pass without funding it."
            currentDiagram = Diagram.MATH
        }
    }

    private fun drawBackground(canvas: Canvas, w: Float, h: Float) {
        val sterile = enlightenment / 100f
        val top = blendColor(Color.rgb(65, 0, 83), Color.rgb(13, 16, 18), sterile)
        val bottom = blendColor(Color.rgb(7, 7, 12), Color.rgb(32, 34, 34), sterile)
        paint.shader = LinearGradient(0f, 0f, 0f, h, top, bottom, Shader.TileMode.CLAMP)
        paint.style = Paint.Style.FILL
        canvas.drawRect(0f, 0f, w, h, paint)
        paint.shader = null

        val pulse = ((sin(SystemClock.uptimeMillis() / 420.0) + 1.0) * 0.5).toFloat()
        val neonAlpha = ((1f - sterile) * (28f + 24f * pulse)).roundToInt()
        paint.color = Color.argb(neonAlpha, 244, 211, 94)
        for (i in 0..5) {
            val y = h * (i + 1) / 7f
            canvas.drawRect(0f, y, w, y + 2f * dp, paint)
        }
    }

    private fun drawHeader(canvas: Canvas, left: Float, right: Float, bottom: Float) {
        paint.style = Paint.Style.FILL
        paint.textAlign = Paint.Align.LEFT
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textSize = 24f * sp
        paint.color = Color.WHITE
        canvas.drawText("Rigged", left, 42f * dp, paint)

        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.NORMAL)
        paint.textSize = 13f * sp
        paint.color = Color.rgb(226, 226, 226)
        canvas.drawText("The Science of Losing", left, 65f * dp, paint)

        val resetRect = RectF(right - 92f * dp, 24f * dp, right, 60f * dp)
        drawButton(canvas, "reset", resetRect, "Reset", false, Color.rgb(64, 64, 68))

        paint.textAlign = Paint.Align.LEFT
        paint.textSize = 14f * sp
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawText("Bankroll ${dollars(bankroll)}", left, 93f * dp, paint)

        paint.textAlign = Paint.Align.RIGHT
        paint.color = if (lastDelta >= 0) Color.rgb(110, 231, 183) else Color.rgb(251, 113, 133)
        canvas.drawText("Last ${dollars(lastDelta)}", right, 93f * dp, paint)

        val meterLeft = left
        val meterRight = right
        val meterTop = bottom - 17f * dp
        val meterRect = RectF(meterLeft, meterTop, meterRight, meterTop + 9f * dp)
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(120, 255, 255, 255)
        canvas.drawRoundRect(meterRect, 4f * dp, 4f * dp, paint)
        val progress = RectF(meterRect.left, meterRect.top, meterRect.left + meterRect.width() * enlightenment / 100f, meterRect.bottom)
        paint.color = blendColor(Color.rgb(244, 211, 94), Color.rgb(180, 180, 180), enlightenment / 100f)
        canvas.drawRoundRect(progress, 4f * dp, 4f * dp, paint)

        paint.textAlign = Paint.Align.CENTER
        paint.textSize = 11f * sp
        paint.color = Color.rgb(222, 222, 222)
        canvas.drawText("Enlightenment $enlightenment%", meterRect.centerX(), meterRect.bottom + 14f * dp, paint)
    }

    private fun drawModeTabs(canvas: Canvas, left: Float, top: Float, right: Float, bottom: Float) {
        val gap = 10f * dp
        val half = (right - left - gap) / 2f
        drawButton(
            canvas,
            "modeSlots",
            RectF(left, top, left + half, bottom),
            "Slots",
            mode == GameMode.SLOTS,
            Color.rgb(190, 24, 93),
        )
        drawButton(
            canvas,
            "modeRoulette",
            RectF(left + half + gap, top, right, bottom),
            "Roulette",
            mode == GameMode.ROULETTE,
            Color.rgb(22, 101, 52),
        )
    }

    private fun drawSlots(canvas: Canvas, rect: RectF) {
        drawPanel(canvas, rect)
        drawPanelTitle(canvas, rect, "SLOTS", "Bet ${dollars(SLOT_BET)} | EV about ${dollars(-120)} per ${dollars(1000)} wagered")

        val reelTop = rect.top + 96f * dp
        val reelBottom = min(rect.bottom - 90f * dp, reelTop + 116f * dp)
        val reelGap = 8f * dp
        val reelWidth = (rect.width() - 48f * dp - reelGap * 2f) / 3f
        var x = rect.left + 24f * dp
        currentSlots.forEachIndexed { index, symbol ->
            val reel = RectF(x, reelTop, x + reelWidth, reelBottom)
            paint.style = Paint.Style.FILL
            paint.color = blendColor(Color.rgb(24, 24, 30), Color.rgb(230, 230, 230), enlightenment / 100f * 0.28f)
            canvas.drawRoundRect(reel, 8f * dp, 8f * dp, paint)
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2f * dp
            paint.color = if (index == 1) Color.rgb(244, 211, 94) else Color.argb(150, 255, 255, 255)
            canvas.drawRoundRect(reel, 8f * dp, 8f * dp, paint)

            paint.style = Paint.Style.FILL
            paint.textAlign = Paint.Align.CENTER
            paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
            paint.textSize = if (symbol.length > 3) 22f * sp else 31f * sp
            paint.color = slotColor(symbol)
            val y = reel.centerY() - (paint.fontMetrics.ascent + paint.fontMetrics.descent) / 2f
            canvas.drawText(symbol, reel.centerX(), y, paint)
            x += reelWidth + reelGap
        }

        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.NORMAL)
        paint.textAlign = Paint.Align.LEFT
        paint.textSize = 13f * sp
        paint.color = Color.rgb(224, 224, 224)
        val payY = reelBottom + 32f * dp
        canvas.drawText("Payouts: 777=${dollars(900)}, triple=${dollars(80)}, pair=${dollars(8)}", rect.left + 24f * dp, payY, paint)
        canvas.drawText("Near 77? Your brain calls it progress. Math calls it Tuesday.", rect.left + 24f * dp, payY + 22f * dp, paint)

        val spinRect = RectF(rect.left + 24f * dp, rect.bottom - 64f * dp, rect.right - 24f * dp, rect.bottom - 18f * dp)
        drawButton(canvas, "slotSpin", spinRect, "Spin ${dollars(SLOT_BET)}", bankroll >= SLOT_BET, Color.rgb(190, 24, 93))
    }

    private fun drawRoulette(canvas: Canvas, rect: RectF) {
        drawPanel(canvas, rect)
        drawPanelTitle(canvas, rect, "ROULETTE", "Bet ${dollars(ROULETTE_BET)} | House edge 5.26%")

        val radius = min(rect.width() * 0.32f, 120f * dp)
        val cx = rect.centerX()
        val cy = rect.top + 160f * dp
        val wheel = RectF(cx - radius, cy - radius, cx + radius, cy + radius)
        val sweep = 360f / 38f
        for (i in 0 until 38) {
            paint.style = Paint.Style.FILL
            paint.color = when (rouletteColorFor(i)) {
                "Red" -> blendColor(Color.rgb(185, 28, 28), Color.rgb(132, 132, 132), enlightenment / 100f)
                "Black" -> blendColor(Color.rgb(18, 18, 20), Color.rgb(74, 74, 74), enlightenment / 100f)
                else -> blendColor(Color.rgb(22, 163, 74), Color.rgb(158, 158, 158), enlightenment / 100f)
            }
            canvas.drawArc(wheel, -90f + i * sweep, sweep + 0.4f, true, paint)
        }

        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f * dp
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawOval(wheel, paint)

        val angle = (-90f + rouletteNumber * sweep + sweep / 2f) * (PI / 180.0)
        paint.style = Paint.Style.FILL
        paint.color = Color.WHITE
        canvas.drawCircle(
            cx + cos(angle).toFloat() * radius * 0.82f,
            cy + sin(angle).toFloat() * radius * 0.82f,
            7f * dp,
            paint,
        )

        paint.textAlign = Paint.Align.CENTER
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textSize = 25f * sp
        paint.color = Color.WHITE
        canvas.drawText(rouletteLabel, cx, cy + 8f * dp, paint)
        paint.textSize = 13f * sp
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.NORMAL)
        paint.color = Color.rgb(224, 224, 224)
        canvas.drawText("Last result: $rouletteColor", cx, cy + radius + 30f * dp, paint)

        val buttonTop = rect.bottom - 118f * dp
        val gap = 10f * dp
        val half = (rect.width() - 48f * dp - gap) / 2f
        drawButton(
            canvas,
            "betRed",
            RectF(rect.left + 24f * dp, buttonTop, rect.left + 24f * dp + half, buttonTop + 42f * dp),
            "Red",
            selectedBet == BetColor.RED,
            Color.rgb(185, 28, 28),
        )
        drawButton(
            canvas,
            "betBlack",
            RectF(rect.left + 24f * dp + half + gap, buttonTop, rect.right - 24f * dp, buttonTop + 42f * dp),
            "Black",
            selectedBet == BetColor.BLACK,
            Color.rgb(31, 41, 55),
        )

        val spinRect = RectF(rect.left + 24f * dp, rect.bottom - 64f * dp, rect.right - 24f * dp, rect.bottom - 18f * dp)
        drawButton(canvas, "rouletteSpin", spinRect, "Spin ${dollars(ROULETTE_BET)}", bankroll >= ROULETTE_BET, Color.rgb(22, 101, 52))
    }

    private fun drawNarrator(canvas: Canvas, rect: RectF) {
        drawPanel(canvas, rect)

        val diagramSize = min(78f * dp, rect.height() - 24f * dp)
        val hasRoomForDiagram = rect.width() > 330f * dp
        val textRight = if (hasRoomForDiagram) rect.right - diagramSize - 30f * dp else rect.right - 16f * dp

        paint.textAlign = Paint.Align.LEFT
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textSize = 15f * sp
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawText(narratorTitle, rect.left + 16f * dp, rect.top + 28f * dp, paint)

        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.NORMAL)
        paint.textSize = 13f * sp
        paint.color = Color.WHITE
        val yAfterLine = drawWrappedText(
            canvas,
            narratorLine,
            rect.left + 16f * dp,
            rect.top + 52f * dp,
            textRight - rect.left - 16f * dp,
            paint,
            18f * dp,
            4,
        )

        paint.textSize = 12f * sp
        paint.color = Color.rgb(198, 198, 198)
        drawWrappedText(
            canvas,
            factLine,
            rect.left + 16f * dp,
            yAfterLine + 12f * dp,
            textRight - rect.left - 16f * dp,
            paint,
            16f * dp,
            3,
        )

        if (hasRoomForDiagram) {
            val diagramRect = RectF(
                rect.right - diagramSize - 16f * dp,
                rect.top + 20f * dp,
                rect.right - 16f * dp,
                rect.top + 20f * dp + diagramSize,
            )
            drawDiagram(canvas, diagramRect, currentDiagram)
        }
    }

    private fun drawEquationOverlay(canvas: Canvas, rect: RectF) {
        val alpha = ((enlightenment - 34).coerceAtLeast(0) * 2.3f).coerceAtMost(140f).roundToInt()
        if (alpha <= 0) return

        paint.style = Paint.Style.FILL
        paint.textAlign = Paint.Align.LEFT
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.NORMAL)
        paint.textSize = 12f * sp
        paint.color = Color.argb(alpha, 230, 230, 230)

        val x = rect.left + 20f * dp
        var y = rect.top + 24f * dp
        listOf(
            "EV = sum(payout * p) - bet",
            "LLN: trials up -> luck washes out",
            "VRRS: random rewards -> persistent behavior",
            "CBT: urge != command",
        ).forEach {
            canvas.drawText(it, x, y, paint)
            y += 18f * dp
        }
    }

    private fun drawPanel(canvas: Canvas, rect: RectF) {
        paint.style = Paint.Style.FILL
        paint.color = blendColor(Color.argb(210, 15, 15, 22), Color.argb(220, 45, 45, 45), enlightenment / 100f)
        canvas.drawRoundRect(rect, 8f * dp, 8f * dp, paint)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 1.5f * dp
        paint.color = blendColor(Color.rgb(244, 211, 94), Color.rgb(160, 160, 160), enlightenment / 100f)
        canvas.drawRoundRect(rect, 8f * dp, 8f * dp, paint)
        paint.style = Paint.Style.FILL
    }

    private fun drawPanelTitle(canvas: Canvas, rect: RectF, title: String, subtitle: String) {
        paint.textAlign = Paint.Align.LEFT
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textSize = 22f * sp
        paint.color = Color.WHITE
        canvas.drawText(title, rect.left + 24f * dp, rect.top + 42f * dp, paint)

        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.NORMAL)
        paint.textSize = 12f * sp
        paint.color = Color.rgb(202, 202, 202)
        canvas.drawText(subtitle, rect.left + 24f * dp, rect.top + 66f * dp, paint)
    }

    private fun drawButton(canvas: Canvas, id: String, rect: RectF, label: String, active: Boolean, baseColor: Int) {
        buttons[id] = RectF(rect)

        val sterile = enlightenment / 100f
        val fill = if (active) baseColor else Color.rgb(58, 58, 64)
        paint.style = Paint.Style.FILL
        paint.color = blendColor(fill, Color.rgb(112, 112, 112), sterile * 0.75f)
        canvas.drawRoundRect(rect, 8f * dp, 8f * dp, paint)

        paint.style = Paint.Style.STROKE
        paint.strokeWidth = if (active) 2f * dp else 1f * dp
        paint.color = if (active) Color.rgb(244, 211, 94) else Color.argb(110, 255, 255, 255)
        canvas.drawRoundRect(rect, 8f * dp, 8f * dp, paint)

        paint.style = Paint.Style.FILL
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = 14f * sp
        val oldSize = paint.textSize
        while (paint.measureText(label) > rect.width() - 14f * dp && paint.textSize > 10f * sp) {
            paint.textSize *= 0.92f
        }
        paint.color = Color.WHITE
        val y = rect.centerY() - (paint.fontMetrics.ascent + paint.fontMetrics.descent) / 2f
        canvas.drawText(label, rect.centerX(), y, paint)
        paint.textSize = oldSize
    }

    private fun drawDiagram(canvas: Canvas, rect: RectF, diagram: Diagram) {
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(48, 255, 255, 255)
        canvas.drawRoundRect(rect, 8f * dp, 8f * dp, paint)

        when (diagram) {
            Diagram.BRAIN -> drawBrainDiagram(canvas, rect)
            Diagram.MATH -> drawMathDiagram(canvas, rect)
            Diagram.SKINNER_BOX -> drawSkinnerDiagram(canvas, rect)
            Diagram.ARCHITECTURE -> drawArchitectureDiagram(canvas, rect)
            Diagram.ROULETTE -> drawRouletteDiagram(canvas, rect)
        }
    }

    private fun drawBrainDiagram(canvas: Canvas, rect: RectF) {
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f * dp
        paint.color = Color.rgb(244, 114, 182)
        val brain = RectF(rect.left + 14f * dp, rect.top + 18f * dp, rect.right - 12f * dp, rect.bottom - 20f * dp)
        canvas.drawOval(brain, paint)
        canvas.drawCircle(brain.left + brain.width() * 0.34f, brain.centerY(), 10f * dp, paint)
        canvas.drawCircle(brain.left + brain.width() * 0.62f, brain.centerY() - 8f * dp, 9f * dp, paint)
        paint.style = Paint.Style.FILL
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawCircle(brain.left + brain.width() * 0.62f, brain.centerY() - 8f * dp, 4f * dp, paint)
    }

    private fun drawMathDiagram(canvas: Canvas, rect: RectF) {
        paint.textAlign = Paint.Align.CENTER
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.MONOSPACE, android.graphics.Typeface.BOLD)
        paint.textSize = 15f * sp
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawText("EV < 0", rect.centerX(), rect.centerY() - 6f * dp, paint)
        paint.textSize = 11f * sp
        paint.color = Color.WHITE
        canvas.drawText("payout - bet", rect.centerX(), rect.centerY() + 18f * dp, paint)
    }

    private fun drawSkinnerDiagram(canvas: Canvas, rect: RectF) {
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f * dp
        paint.color = Color.WHITE
        val box = RectF(rect.left + 16f * dp, rect.top + 18f * dp, rect.right - 16f * dp, rect.bottom - 18f * dp)
        canvas.drawRoundRect(box, 5f * dp, 5f * dp, paint)
        paint.style = Paint.Style.FILL
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawRect(box.right - 20f * dp, box.centerY() - 4f * dp, box.right - 8f * dp, box.centerY() + 4f * dp, paint)
        canvas.drawCircle(box.left + 22f * dp, box.bottom - 18f * dp, 6f * dp, paint)
    }

    private fun drawArchitectureDiagram(canvas: Canvas, rect: RectF) {
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f * dp
        paint.color = Color.rgb(244, 211, 94)
        val building = RectF(rect.left + 18f * dp, rect.top + 20f * dp, rect.right - 18f * dp, rect.bottom - 14f * dp)
        canvas.drawRect(building, paint)
        paint.textAlign = Paint.Align.CENTER
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.SANS_SERIF, android.graphics.Typeface.BOLD)
        paint.textSize = 11f * sp
        paint.color = Color.WHITE
        canvas.drawText("NO", building.centerX(), building.centerY() - 4f * dp, paint)
        canvas.drawText("CLOCKS", building.centerX(), building.centerY() + 12f * dp, paint)
    }

    private fun drawRouletteDiagram(canvas: Canvas, rect: RectF) {
        val radius = min(rect.width(), rect.height()) * 0.33f
        val cx = rect.centerX()
        val cy = rect.centerY()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2f * dp
        paint.color = Color.rgb(244, 211, 94)
        canvas.drawCircle(cx, cy, radius, paint)
        paint.style = Paint.Style.FILL
        paint.color = Color.rgb(22, 163, 74)
        canvas.drawCircle(cx, cy - radius, 5f * dp, paint)
        canvas.drawCircle(cx + radius * 0.7f, cy - radius * 0.7f, 5f * dp, paint)
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = 11f * sp
        paint.color = Color.WHITE
        canvas.drawText("0/00", cx, cy + 4f * dp, paint)
    }

    private fun drawWrappedText(
        canvas: Canvas,
        text: String,
        x: Float,
        y: Float,
        maxWidth: Float,
        textPaint: Paint,
        lineHeight: Float,
        maxLines: Int,
    ): Float {
        val words = text.split(" ")
        var line = ""
        var cursorY = y
        var lines = 0
        for (word in words) {
            val candidate = if (line.isEmpty()) word else "$line $word"
            if (textPaint.measureText(candidate) <= maxWidth || line.isEmpty()) {
                line = candidate
            } else {
                canvas.drawText(line, x, cursorY, textPaint)
                lines += 1
                cursorY += lineHeight
                if (lines >= maxLines) return cursorY
                line = word
            }
        }
        if (line.isNotEmpty() && lines < maxLines) {
            canvas.drawText(line, x, cursorY, textPaint)
            cursorY += lineHeight
        }
        return cursorY
    }

    private fun rouletteColorFor(number: Int): String {
        return when {
            number == 0 || number == 37 -> "Green"
            number in rouletteRedNumbers -> "Red"
            else -> "Black"
        }
    }

    private fun slotColor(symbol: String): Int {
        val sterile = enlightenment / 100f
        val color = when (symbol) {
            "7" -> Color.rgb(248, 113, 113)
            "BAR" -> Color.rgb(244, 211, 94)
            "CH" -> Color.rgb(244, 63, 94)
            "BELL" -> Color.rgb(251, 191, 36)
            "EV" -> Color.rgb(125, 211, 252)
            else -> Color.rgb(196, 181, 253)
        }
        return blendColor(color, Color.rgb(214, 214, 214), sterile)
    }

    private fun blendColor(from: Int, to: Int, t: Float): Int {
        val f = t.coerceIn(0f, 1f)
        return Color.argb(
            lerp(Color.alpha(from), Color.alpha(to), f),
            lerp(Color.red(from), Color.red(to), f),
            lerp(Color.green(from), Color.green(to), f),
            lerp(Color.blue(from), Color.blue(to), f),
        )
    }

    private fun lerp(start: Int, end: Int, t: Float): Int {
        return (start + (end - start) * t).roundToInt().coerceIn(0, 255)
    }

    private fun dollars(amount: Int): String {
        return if (amount < 0) "-\$${abs(amount)}" else "\$$amount"
    }

    private companion object {
        const val SLOT_BET = 10
        const val ROULETTE_BET = 20
    }
}
