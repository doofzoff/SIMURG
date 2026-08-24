# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# Runnable quickstart — needs no LLM and no network. Two scenarios:
#   A) corruption from the very start  → held before release → ZERO leak
#   B) a healthy answer that derails    → detected with a short latency, rest aborted
# Run:  python3 examples/quickstart.py
# ═══════════════════════════════════════════════════════════════════════════════
from simurg import Simurg, CORRUPT


def feed_stream(text, size=24):
    """Return (text_shown_to_user, verdict, onset_char)."""
    s = Simurg()
    shown = []
    for i in range(0, len(text), size):
        v = s.feed(text[i:i + size])
        if v.released:
            shown.append(v.released)
        if v.state == CORRUPT:
            return "".join(shown), v, v.onset_char
    final = s.finish()
    if final.released:
        shown.append(final.released)
    return "".join(shown), final, None


CLEAN = ("The non-oil output gap narrows to -0.4 percent this quarter from a "
         "-1.6 percent baseline, while headline inflation stays inside the target "
         "corridor at about 5.7 percent. Credit conditions ease modestly and the "
         "current account surplus holds near 7 percent of GDP. Construction and "
         "transport lead the recovery, with services close behind. ")
GARBAGE = "#REF! -0.00 -0.00 MCI 0.00 #REF! -0.00 0.00 " * 30
LOOP = "budget expenditure rises budget expenditure rises " * 40


def main():
    print("── Scenario A: corrupt from the first token ──")
    shown, v, onset = feed_stream(GARBAGE)
    print("verdict :", v.state)
    print("reason  :", "; ".join(v.reasons))
    print(f"shown to user : {len(shown)} chars  →  "
          f"{'ZERO LEAK ✅' if len(shown) == 0 else 'leaked ' + str(len(shown)) + ' chars ❌'}")

    print("\n── Scenario B: a healthy answer that derails into a loop ──")
    shown, v, onset = feed_stream(CLEAN + LOOP)
    print("verdict :", v.state)
    print("reason  :", "; ".join(v.reasons))
    print(f"clean answer was ~{len(CLEAN)} chars; derail began there.")
    print(f"SIMURG aborted at char {onset}; the loop was cut off after a short "
          f"detection latency\ninstead of streaming forever. Everything past the "
          f"abort never reaches the user.")


if __name__ == "__main__":
    main()
