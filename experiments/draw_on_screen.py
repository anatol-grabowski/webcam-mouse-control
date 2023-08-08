import gi
import threading
import time

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")

from gi.repository import Gdk  # noqa
from gi.repository import Gtk  # noqa

CSS = b"""
#toplevel {
    background-color: rgba(0, 0, 0, 0);
}
"""

style_provider = Gtk.CssProvider()
style_provider.load_from_data(CSS)

Gtk.StyleContext.add_provider_for_screen(
    Gdk.Screen.get_default(),
    style_provider,
    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
)


class Crosshair(Gtk.DrawingArea):
    def __init__(self):
        super(Crosshair, self).__init__()

    def do_draw(self, cr):
        # Get the dimensions of the drawing area
        width = self.get_allocated_width()
        height = self.get_allocated_height()

        # Calculate the center of the drawing area
        center_x = width / 2
        center_y = height / 2

        # Set up colors
        blue_color = Gdk.RGBA(0.0, 0.0, 1.0, 1.0)
        yellow_color = Gdk.RGBA(1.0, 1.0, 0.0, 1.0)

        # Draw the blue crosshair
        cr.set_source_rgba(*blue_color)
        cr.set_line_width(2.0)
        cr.move_to(center_x, 0)
        cr.line_to(center_x, height)
        cr.move_to(0, center_y)
        cr.line_to(width, center_y)
        cr.stroke()

        # Draw the yellow outline
        cr.set_source_rgba(*yellow_color)
        cr.set_line_width(4.0)
        cr.arc(center_x, center_y, 50, 0, 2 * 3.1415)
        cr.stroke()


button1 = Gtk.Button(label="Hello, world!")
box = Gtk.Box(spacing=50)
# box.pack_start(button1, True, True, 50)

crosshair_widget = Crosshair()
box.pack_start(crosshair_widget, True, True, 0)

window = Gtk.Window(title="Hello World", name="toplevel")
window.set_visual(window.get_screen().get_rgba_visual())
window.set_decorated(False)
window.set_keep_above(True)
window.set_accept_focus(False)
window.add(box)
window.show_all()
window.move(10, 20)
window.connect("destroy", Gtk.main_quit)


thread = threading.Thread(target=Gtk.main, daemon=True, args=[])
thread.start()

time.sleep(2)
window.move(-20, -20)
time.sleep(10)
