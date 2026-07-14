from localflow.overlay import RecordingOverlay, WaveModel, overlay_frame

SCREEN = (0, 0, 1440, 900)


def test_overlay_frame_bottom_center():
    x, y = overlay_frame(SCREEN, (150, 32), "bottom-center", margin=60)
    assert x == (1440 - 150) / 2
    assert y == 60


def test_overlay_frame_top_right():
    x, y = overlay_frame(SCREEN, (150, 32), "top-right")
    assert x == 1440 - 150 - 24
    assert y == 900 - 32 - 36


def test_overlay_frame_respects_screen_origin():
    x, y = overlay_frame((100, 50, 1440, 900), (150, 32), "bottom-center", margin=60)
    assert x == 100 + (1440 - 150) / 2
    assert y == 50 + 60


def test_wave_amplitude_rises_smoothly_toward_target():
    model = WaveModel(attack=0.5)
    model.feed(9.9)  # loud → target clipped to 1.0
    assert model.amplitude == 0.0  # feed alone must not jump the animation
    model.tick()
    first = model.amplitude
    model.tick()
    assert 0 < first < 1.0
    assert first < model.amplitude < 1.0  # approaches, never overshoots


def test_wave_amplitude_decays_slowly_after_silence():
    model = WaveModel()
    model.feed(9.9)
    for _ in range(30):
        model.tick()
    loud = model.amplitude
    model.feed(0.0)  # silence
    model.tick()
    assert 0 < model.amplitude < loud  # decays gradually, no snap to zero


def test_phase_advances_every_tick_so_the_wave_flows():
    model = WaveModel()
    p0 = model.phase
    model.tick()
    model.tick()
    assert model.phase > p0
    a = model.points(100, 32)
    model.tick()
    b = model.points(100, 32)
    assert a != b  # same amplitude, different phase → visibly moving


def test_points_taper_to_the_midline_at_both_ends():
    model = WaveModel()
    model.feed(9.9)
    for _ in range(50):
        model.tick()
    pts = model.points(100, 32, n=50)
    mid = 16.0
    assert len(pts) == 50
    assert abs(pts[0][1] - mid) < 0.01
    assert abs(pts[-1][1] - mid) < 0.01
    assert max(abs(y - mid) for _, y in pts) > 3  # but it does swing in between


def test_points_stay_inside_the_box():
    model = WaveModel()
    model.feed(9.9)
    for _ in range(100):
        model.tick()
        for x, y in model.points(150, 32):
            assert 0 <= x <= 150
            assert 0 < y < 32


def test_reset_clears_motion_state():
    model = WaveModel()
    model.feed(0.5)
    model.tick()
    model.reset()
    assert model.amplitude == 0.0
    assert model.phase == 0.0


def make_overlay(**kwargs):
    dispatched = []
    overlay = RecordingOverlay(dispatch=lambda fn, *a: dispatched.append(fn.__name__),
                               **kwargs)
    return overlay, dispatched


def test_show_hide_toggle_visibility_and_dispatch():
    overlay, dispatched = make_overlay()
    assert overlay.visible is False
    overlay.show()
    assert overlay.visible is True
    overlay.hide()
    assert overlay.visible is False
    assert dispatched == ["_show_main", "_hide_main"]


def test_feed_only_registers_while_visible():
    overlay, _ = make_overlay()
    overlay.feed(0.5)  # hidden — ignored
    assert overlay.model._target == 0.0
    overlay.show()
    overlay.feed(0.5)
    assert overlay.model._target == 1.0  # 0.5 rms is well above the ceiling


def test_show_resets_previous_session_wave():
    overlay, _ = make_overlay()
    overlay.show()
    overlay.feed(0.9)
    overlay.model.tick()
    overlay.hide()
    overlay.show()
    assert overlay.model.amplitude == 0.0


def test_invalid_position_falls_back_to_default():
    overlay, _ = make_overlay(position="upside-down")
    assert overlay.position == "bottom-center"
