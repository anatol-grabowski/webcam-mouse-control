
import gi
import cv2
import cv2
import win32api

cv2.namedWindow("my_window")
cv2.setMouseCallback("my_window", mouse_evt)


def mouse_evt(event, x, y, flags, param):
    # could also probably do: def mouse_evt(*args):
    win32api.SetCursor(None)

# Create a GTK window to set the cursor
# window = Gtk.Window()
# window.set_default_size(800, 600)


# Load and show an image using cv2
image = cv2.imread('/home/anatoly/_tot/proj/ml/eye_controlled_mouse/data/v16/1-1 [5 5] 1691931338468.jpeg')
cv2.imshow('Image', image)
while True:
    cv2.waitKey(1)  # Show the OpenCV window

# Set the cursor to an hourglass (wait) cursor
# cursor = Gdk.Cursor.new_for_name(Gdk.CursorType.WAIT)
# window.get_window().set_cursor(cursor)

# Run the GTK main loop
# Gtk.main()
