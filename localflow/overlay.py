"""On-screen recording indicator: a floating, flowing waveform (Wispr-style).

A small borderless, click-through, always-on-top panel that appears while
recording. It renders a continuous wave whose amplitude follows the (smoothed)
microphone level and whose phase advances every frame, so the wave visibly
flows instead of jittering. All AppKit work happens lazily on the main thread;
the wave math lives in the pure-Python WaveModel so it is testable headless.
"""

import math

POSITIONS = ("bottom-center", "top-right")


def overlay_frame(screen, size, position="bottom-center", margin=60):
    """Origin (x, y) for the panel on a screen rect (x, y, w, h).

    Cocoa's y axis grows upward from the bottom-left of the screen.
    """
    sx, sy, sw, sh = screen
    w, h = size
    if position == "top-right":
        return (sx + sw - w - 24, sy + sh - h - 36)
    return (sx + (sw - w) / 2, sy + margin)


class WaveModel:
    """Smoothed mic amplitude + advancing phase → points of a flowing wave.

    feed() may be called from the audio thread (sets the target only);
    tick() advances the animation one frame and belongs to the render timer.
    """

    IDLE_RIPPLE = 0.08  # visible baseline motion even in silence

    def __init__(self, floor=0.004, ceil=0.22, attack=0.45, decay=0.10, speed=0.45):
        self.floor = floor
        self.ceil = ceil
        self.attack = attack
        self.decay = decay
        self.speed = speed
        self.amplitude = 0.0
        self.phase = 0.0
        self._target = 0.0

    def reset(self):
        self.amplitude = 0.0
        self.phase = 0.0
        self._target = 0.0

    def feed(self, level):
        level = max(0.0, float(level))
        if level <= self.floor:
            self._target = 0.0
        else:
            self._target = min(1.0, (level - self.floor) / (self.ceil - self.floor))

    def tick(self):
        delta = self._target - self.amplitude
        self.amplitude += delta * (self.attack if delta > 0 else self.decay)
        self.phase += self.speed

    def points(self, width, height, n=64):
        """Polyline of the wave inside a width×height box."""
        mid = height / 2.0
        max_amp = mid - 3.0
        amp = self.IDLE_RIPPLE + (1.0 - self.IDLE_RIPPLE) * self.amplitude
        pts = []
        for i in range(n):
            t = i / (n - 1)
            envelope = math.sin(math.pi * t)  # pinch the wave at both ends
            swing = (0.72 * math.sin(2 * math.pi * 2.2 * t - self.phase)
                     + 0.28 * math.sin(2 * math.pi * 4.3 * t - self.phase * 1.6))
            pts.append((t * width, mid + max_amp * amp * envelope * swing))
        return pts


_WaveViewClass = None


def _wave_view_class():
    """Create the NSView subclass once (ObjC class names must be unique)."""
    global _WaveViewClass
    if _WaveViewClass is None:
        from AppKit import NSBezierPath, NSColor, NSView
        from Foundation import NSMakePoint

        class LocalFlowWaveView(NSView):
            model = None  # set after alloc/init

            def drawRect_(self, rect):
                bounds = self.bounds()
                pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    bounds, bounds.size.height / 2, bounds.size.height / 2)
                NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.8).setFill()
                pill.fill()
                if self.model is None:
                    return
                inset = bounds.size.height / 2.0
                pts = self.model.points(bounds.size.width - 2 * inset,
                                        bounds.size.height)
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(inset + pts[0][0], pts[0][1]))
                for x, y in pts[1:]:
                    path.lineToPoint_(NSMakePoint(inset + x, y))
                path.setLineJoinStyle_(1)  # round joins
                glow = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.35, 0.78, 1.0, 0.25)
                core = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.45, 0.82, 1.0, 0.95)
                glow.setStroke()
                path.setLineWidth_(4.0)
                path.stroke()
                core.setStroke()
                path.setLineWidth_(1.8)
                path.stroke()

        _WaveViewClass = LocalFlowWaveView
    return _WaveViewClass


class RecordingOverlay:
    WIDTH = 150
    HEIGHT = 32
    FPS = 30.0

    def __init__(self, position="bottom-center", dispatch=None):
        self.position = position if position in POSITIONS else "bottom-center"
        self.model = WaveModel()
        self.visible = False
        self._dispatch = dispatch or self._main_thread_dispatch
        self._panel = None
        self._view = None
        self._timer = None

    @staticmethod
    def _main_thread_dispatch(fn, *args):
        from PyObjCTools import AppHelper

        AppHelper.callAfter(fn, *args)

    # -- thread-safe API ----------------------------------------------------
    def show(self):
        self.model.reset()
        self.visible = True
        self._dispatch(self._show_main)

    def hide(self):
        self.visible = False
        self._dispatch(self._hide_main)

    def feed(self, level):
        if self.visible:
            self.model.feed(level)  # target only; the render timer animates

    # -- main-thread implementation -------------------------------------------
    def _ensure_panel(self):
        if self._panel is not None:
            return
        from AppKit import (
            NSBackingStoreBuffered,
            NSColor,
            NSPanel,
            NSScreen,
            NSStatusWindowLevel,
            NSWindowStyleMaskBorderless,
            NSWindowStyleMaskNonactivatingPanel,
        )
        from Foundation import NSMakeRect

        screen = NSScreen.mainScreen()
        sf = screen.visibleFrame()
        x, y = overlay_frame((sf.origin.x, sf.origin.y, sf.size.width, sf.size.height),
                             (self.WIDTH, self.HEIGHT), self.position)
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, self.WIDTH, self.HEIGHT),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)
        panel.setHasShadow_(False)
        # canJoinAllSpaces | stationary | fullScreenAuxiliary
        panel.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))
        view = _wave_view_class().alloc().initWithFrame_(
            NSMakeRect(0, 0, self.WIDTH, self.HEIGHT))
        view.model = self.model
        panel.setContentView_(view)
        self._panel = panel
        self._view = view

    def _show_main(self):
        try:
            self._ensure_panel()
            self._panel.orderFrontRegardless()
            self._start_timer()
        except Exception:
            self._panel = None  # AppKit unavailable — degrade silently

    def _hide_main(self):
        self._stop_timer()
        if self._panel is not None:
            self._panel.orderOut_(None)

    def _start_timer(self):
        if self._timer is not None:
            return
        from Foundation import NSTimer

        def _frame(_timer):
            self.model.tick()
            if self._view is not None:
                self._view.setNeedsDisplay_(True)

        self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            1.0 / self.FPS, True, _frame)

    def _stop_timer(self):
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
