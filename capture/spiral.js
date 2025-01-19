
function spiral(xmin, ymin, xmax, ymax, xsteps, ysteps) {
    const pointsList = [];
    let x = 0, y = 0;
    const numPoints = (xsteps + 1) * (ysteps + 1);
    let dir = 'right';
    let x0 = 0, y0 = 0, x1 = xsteps, y1 = ysteps;

    while (pointsList.length < numPoints) {
        pointsList.push([x, y]);
        if (dir === 'right') {
            x += 1;
            if (x === x1) {
                y0 += 1;
                dir = 'down';
            }
            continue;
        }
        if (dir === 'down') {
            y += 1;
            if (y === y1) {
                x1 -= 1;
                dir = 'left';
            }
            continue;
        }
        if (dir === 'left') {
            x -= 1;
            if (x === x0) {
                y1 -= 1;
                dir = 'up';
            }
            continue;
        }
        if (dir === 'up') {
            y -= 1;
            if (y === y0) {
                x0 += 1;
                dir = 'right';
            }
            continue;
        }
    }

    const dx = (xmax - xmin) / xsteps;
    const dy = (ymax - ymin) / ysteps;
    const points = pointsList.map(([px, py]) => [
        xmin + px * dx,
        ymin + py * dy
    ]);

    return points;
}

window.spiral = spiral