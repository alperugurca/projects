// script.js
// ------------------------------------------------------------
// Step 1 – Define the Calculator class
// ------------------------------------------------------------

class Calculator {
  // Private fields (ES2022 syntax)
  #currentValue = "0"; // what is shown on the display – always a string
  #previousValue = null; // numeric value stored before an operator is chosen
  #operator = null; // '+', '-', '*', '/' or null
  #shouldResetDisplay = false; // true after an operator is chosen or after a result is shown
  #error = false; // true when an error (e.g., division by zero) occurs

  // ----------------------------------------------------------
  // Append a digit (as a string) to the current value.
  // ----------------------------------------------------------
  appendNumber(num) {
    // If we are in an error state, ignore further numeric input.
    if (this.#error) return;

    // If the display should be reset (after an operator or a result), start fresh.
    if (this.#shouldResetDisplay) {
      this.#currentValue = "";
      this.#shouldResetDisplay = false;
    }

    // Prevent leading multiple zeros (e.g., "00").
    if (this.#currentValue === "0" && num === "0") {
      return; // ignore extra leading zeros
    }

    // If the current value is just "0" and a non‑zero digit is pressed, replace it.
    if (this.#currentValue === "0" && num !== "0") {
      this.#currentValue = num;
    } else {
      this.#currentValue += num;
    }
  }

  // ----------------------------------------------------------
  // Choose an operator (+, -, *, /). Stores the current value as
  // the previous value and prepares the calculator for the next
  // number entry.
  // ----------------------------------------------------------
  chooseOperator(op) {
    if (this.#error) return; // ignore if we are in an error state

    // If there is already a pending operation, compute it first.
    if (this.#operator && this.#previousValue !== null && !this.#shouldResetDisplay) {
      // Compute intermediate result so chained operations work like a real calculator.
      const intermediate = this.compute();
      if (intermediate === "Error") {
        // compute already set error flag; abort further processing.
        return;
      }
    }

    // Store the current number as the previous value.
    this.#previousValue = Number(this.#currentValue);
    this.#operator = op;
    this.#shouldResetDisplay = true; // next digit should start a new number
  }

  // ----------------------------------------------------------
  // Perform the pending calculation and return the result.
  // Returns a number (as a string) or the string "Error".
  // ----------------------------------------------------------
  compute() {
    if (this.#error) return "Error"; // propagate existing error

    if (this.#operator === null || this.#previousValue === null) {
      // Nothing to compute – just return the current display value.
      return this.#currentValue;
    }

    const current = Number(this.#currentValue);
    let result;

    switch (this.#operator) {
      case "+":
        result = this.#previousValue + current;
        break;
      case "-":
        result = this.#previousValue - current;
        break;
      case "*":
        result = this.#previousValue * current;
        break;
      case "/":
        if (current === 0) {
          // Division by zero – set error state.
          this.#error = true;
          this.#currentValue = "Error";
          this.#previousValue = null;
          this.#operator = null;
          this.#shouldResetDisplay = true;
          return "Error";
        }
        result = this.#previousValue / current;
        break;
      default:
        // Unknown operator – treat as no‑op.
        result = current;
    }

    // Store the result as the new current value.
    const resultStr = Number.isInteger(result) ? String(result) : String(result);
    this.#currentValue = resultStr;
    // Reset operator state – ready for a new operation.
    this.#operator = null;
    this.#previousValue = null;
    this.#shouldResetDisplay = true;
    return resultStr;
  }

  // ----------------------------------------------------------
  // Reset the calculator to its initial state.
  // ----------------------------------------------------------
  clear() {
    this.#currentValue = "0";
    this.#previousValue = null;
    this.#operator = null;
    this.#shouldResetDisplay = false;
    this.#error = false;
  }

  // ----------------------------------------------------------
  // Return the value that should be shown on the screen.
  // ----------------------------------------------------------
  getDisplayValue() {
    return this.#currentValue;
  }
}

// Export the class to the global scope so other scripts (or tests) can use it.
window.Calculator = Calculator;

// ------------------------------------------------------------
// Step 2 – Instantiate Calculator and cache DOM elements
// ------------------------------------------------------------
const calculator = new Calculator();
const display = document.getElementById("display");
const digitButtons = document.querySelectorAll(".btn.digit");
const operatorButtons = document.querySelectorAll(".btn.operator");
const equalsButton = document.getElementById("equals");
const clearButton = document.getElementById("clear");

// ------------------------------------------------------------
// Helper – update the visual display
// ------------------------------------------------------------
function updateDisplay() {
  display.textContent = calculator.getDisplayValue();
}

// ------------------------------------------------------------
// Step 3 – UI Event Handlers
// ------------------------------------------------------------
// Digit buttons
if (digitButtons) {
  digitButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const value = e.target.dataset.value;
      calculator.appendNumber(value);
      updateDisplay();
    });
  });
}

// Operator buttons
if (operatorButtons) {
  operatorButtons.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const op = e.target.dataset.operator;
      calculator.chooseOperator(op);
      // No immediate display change – the display still shows the previous number.
      updateDisplay();
    });
  });
}

// Equals button
if (equalsButton) {
  equalsButton.addEventListener("click", () => {
    const result = calculator.compute();
    display.textContent = result;
  });
}

// Clear button
if (clearButton) {
  clearButton.addEventListener("click", () => {
    calculator.clear();
    updateDisplay();
  });
}

// ------------------------------------------------------------
// Step 4 – Keyboard Support
// ------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  const key = e.key;
  // Digits 0‑9
  if (/^[0-9]$/.test(key)) {
    e.preventDefault();
    calculator.appendNumber(key);
    updateDisplay();
    return;
  }

  // Operators
  if (["+", "-", "*", "/"].includes(key)) {
    e.preventDefault();
    calculator.chooseOperator(key);
    updateDisplay();
    return;
  }

  // Equals / Enter
  if (key === "Enter" || key === "=") {
    e.preventDefault();
    const result = calculator.compute();
    display.textContent = result;
    return;
  }

  // Clear / Escape / C
  if (key === "Escape" || key === "c" || key === "C") {
    e.preventDefault();
    calculator.clear();
    updateDisplay();
    return;
  }

  // Backspace – optional simple implementation (remove last digit)
  if (key === "Backspace") {
    e.preventDefault();
    // Simple backspace handling: remove last character unless we are showing an error.
    if (display.textContent !== "Error") {
      const current = calculator.getDisplayValue();
      if (current.length > 1) {
        const newVal = current.slice(0, -1);
        // Reset calculator state and rebuild the value.
        calculator.clear();
        for (const ch of newVal) {
          calculator.appendNumber(ch);
        }
      } else {
        // If only one digit left, reset to 0.
        calculator.clear();
      }
      updateDisplay();
    }
    return;
  }
});

// Initial display sync (in case the HTML default differs).
updateDisplay();
