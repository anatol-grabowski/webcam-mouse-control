create a single file html+js+css app that has:
- a select for camera selection - should allow selection among available cameras
- a select for camera mode (resolution and framerate) - should allow selection among modes available for selected camera
- a button to go fullscreen
- start/stop capture button - should start a recording into file
- download capture button - should allow to download 2 files recorded video and points json (more about points below)
- a button to regenerate path (more about path later), if the window is resized then the path should be generated on the whole window
- on the background it should show an image from the selected camera (flipped horizontally) - the image should be centered on the screen and scaled to fit available space, gray bars of background on left/right or top/bottom are acceptable if the aspect ratio of the video doesn't match the aspect ratio of the window/screen
Keep in mind that the permissions to use camera should be requested before listing available devices

When the app is opened it should generate a random path on the screen and draw it above the video. A starting point on the path should be highlighted by a green crosshair.
- j key - goes to next point - smoothly moved the crosshair along the path (1px at a time, make sure the crosshair doesn't just jump to the next point, the position should be interpolated between current and next point)
- k key - goes to the previous point
- space key - starts and stops the recording/capture
- f key - toggles fullscreen mode

The background (video from camera) and the path should take the whole window. The buttons should be displayed over the background.
When the recording is started for each frame of the video x and y of the crosshair in screen (not window or client area!) coordinates should be recorded. Keep in mind that j and k can move the cursor during the recording. For each point in points json there should be x, y, frame number (starting from 0), timestamp in seconds from the start of the recording (not from when the camera started the translation). Points json also should have info about selected camera, camera mode and the date and time of when the recording was started