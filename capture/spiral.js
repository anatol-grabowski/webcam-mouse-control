function spiral(xmin, ymin, xmax, ymax, xsteps, ysteps, startCorner = 0, direction = 0) {
    const pointsList = [];
    let x, y;
    let dir;
    const numPoints = (xsteps + 1) * (ysteps + 1);
    let x0 = 0, y0 = 0, x1 = xsteps, y1 = ysteps;
    startCorner = {
        0: 'top-left',
        1: 'top-right',
        2: 'bottom-right',
        3: 'bottom-left',
    }[startCorner]

    // Determine initial position and direction based on starting corner and direction
    switch (startCorner) {
        case 'top-left':
            x = 0;
            y = 0;
            dir = direction === 0 ? 'right' : 'down';
            break;
        case 'top-right':
            x = xsteps;
            y = 0;
            dir = direction === 0 ? 'down' : 'left';
            break;
        case 'bottom-right':
            x = xsteps;
            y = ysteps;
            dir = direction === 0 ? 'left' : 'up';
            break;
        case 'bottom-left':
            x = 0;
            y = ysteps;
            dir = direction === 0 ? 'up' : 'right';
            break;
        default:
            throw new Error('Invalid startCorner. Must be "top-left", "top-right", "bottom-left", or "bottom-right".');
    }

    // Generate the spiral
    while (pointsList.length < numPoints) {
        pointsList.push([x, y]);
        if (dir === 'right') {
            x += 1;
            if (x === x1) {
                if (direction === 0) {
                    y0 += 1; // Update bounds clockwise
                } else {
                    y1 -= 1; // Update bounds counterclockwise
                }
                dir = direction === 0 ? 'down' : 'up';
            }
            continue;
        }
        if (dir === 'down') {
            y += 1;
            if (y === y1) {
                if (direction === 0) {
                    x1 -= 1; // Update bounds clockwise
                } else {
                    x0 += 1; // Update bounds counterclockwise
                }
                dir = direction === 0 ? 'left' : 'right';
            }
            continue;
        }
        if (dir === 'left') {
            x -= 1;
            if (x === x0) {
                if (direction === 0) {
                    y1 -= 1; // Update bounds clockwise
                } else {
                    y0 += 1; // Update bounds counterclockwise
                }
                dir = direction === 0 ? 'up' : 'down';
            }
            continue;
        }
        if (dir === 'up') {
            y -= 1;
            if (y === y0) {
                if (direction === 0) {
                    x0 += 1; // Update bounds clockwise
                } else {
                    x1 -= 1; // Update bounds counterclockwise
                }
                dir = direction === 0 ? 'right' : 'left';
            }
            continue;
        }
    }

    // Scale points to the given xmin, ymin, xmax, ymax
    const dx = (xmax - xmin) / xsteps;
    const dy = (ymax - ymin) / ysteps;
    const points = pointsList.map(([px, py]) => [
        xmin + px * dx,
        ymin + py * dy
    ]);

    return points;
}

// Example usage:
window.spiral = spiral;

