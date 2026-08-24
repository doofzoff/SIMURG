"""core sentinel regressions: clean streams stay clean, corrupt streams abort."""
from simurg.detection.sentinel import Simurg

CLEAN = ("The quarterly report showed steady growth across all regions. "
         "Revenue increased by eleven percent year over year, driven mostly "
         "by enterprise subscriptions. Operating expenses rose more slowly, "
         "so margins expanded for the third consecutive quarter. Management "
         "expects this trend to continue into the next fiscal year, provided "
         "that input costs remain stable and hiring plans stay on track.")


def _run(text):
    s = Simurg()
    for i in range(0, len(text), 40):
        v = s.feed(text[i:i + 40])
    return s.finish()


def test_clean_text_is_clean():
    v = _run(CLEAN * 3)
    assert v.state == "clean"
    assert v.p_corrupt < 0.5


def test_repetition_loop_aborts():
    stream = CLEAN + "the same phrase keeps repeating in a tight loop " * 40
    v = _run(stream)
    assert v.state == "corrupt"
    assert v.reasons


def test_onset_localization_near_true_onset():
    onset = len(CLEAN)
    stream = CLEAN + "the same phrase keeps repeating in a tight loop " * 40
    v = _run(stream)
    assert v.onset_char is not None
    assert abs(v.onset_char - onset) < 800


def test_zero_leak_when_corrupt_from_start():
    stream = "the same phrase keeps repeating in a tight loop " * 60
    s = Simurg()
    released = ""
    for i in range(0, len(stream), 40):
        v = s.feed(stream[i:i + 40])
        released += v.released
        if v.state == "corrupt":
            break
    assert v.state == "corrupt"
    assert released == ""          # hold window: nothing reached the UI


def test_structural_garbage_aborts():
    stream = CLEAN + " #REF! -0.00 -0.00 #REF! -0.00 0.00 #REF! -0.00 " * 25
    v = _run(stream)
    assert v.state == "corrupt"
