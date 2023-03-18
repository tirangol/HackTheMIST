class Vector2 {
	constructor(x, y) {
		this.x = x;
		this.y = y;
	}

	dot(other) {
		return this.x*other.x + this.y*other.y;
	}
}

function Shuffle(arrayToShuffle) {
	for(let e = arrayToShuffle.length-1; e > 0; e--) {
		const index = Math.round(Math.random()*(e-1));
		const temp = arrayToShuffle[e];
		
		arrayToShuffle[e] = arrayToShuffle[index];
		arrayToShuffle[index] = temp;
	}
}

function MakePermutation() {
	const permutation = [];
	for(let i = 0; i < 256; i++) {
		permutation.push(i);
	}

	Shuffle(permutation);
	
	for(let i = 0; i < 256; i++) {
		permutation.push(permutation[i]);
	}
	
	return permutation;
}

function GetConstantVector(v) {
	// v is the value from the permutation table
	const h = v & 3;
	if(h == 0)
		return new Vector2(1.0, 1.0);
	else if(h == 1)
		return new Vector2(-1.0, 1.0);
	else if(h == 2)
		return new Vector2(-1.0, -1.0);
	else
		return new Vector2(1.0, -1.0);
}

function Fade(t) {
	return ((6*t - 15)*t + 10)*t*t*t;
}

function Lerp(t, a1, a2) {
	return a1 + t*(a2-a1);
}

var Permutation = MakePermutation();
function Noise2D(x, y) {
	const X = Math.floor(x) & 255;
	const Y = Math.floor(y) & 255;

	const xf = x-Math.floor(x);
	const yf = y-Math.floor(y);

	const topRight = new Vector2(xf-1.0, yf-1.0);
	const topLeft = new Vector2(xf, yf-1.0);
	const bottomRight = new Vector2(xf-1.0, yf);
	const bottomLeft = new Vector2(xf, yf);
	
	// Select a value from the permutation array for each of the 4 corners
	const valueTopRight = Permutation[Permutation[X+1]+Y+1];
	const valueTopLeft = Permutation[Permutation[X]+Y+1];
	const valueBottomRight = Permutation[Permutation[X+1]+Y];
	const valueBottomLeft = Permutation[Permutation[X]+Y];
	
	const dotTopRight = topRight.dot(GetConstantVector(valueTopRight));
	const dotTopLeft = topLeft.dot(GetConstantVector(valueTopLeft));
	const dotBottomRight = bottomRight.dot(GetConstantVector(valueBottomRight));
	const dotBottomLeft = bottomLeft.dot(GetConstantVector(valueBottomLeft));
	
	const u = Fade(xf);
	const v = Fade(yf);
	
	return Lerp(u,
		Lerp(v, dotBottomLeft, dotTopLeft),
		Lerp(v, dotBottomRight, dotTopRight)
	);
}


var canvas;
var ctx;
var canvas_width;
var canvas_height;

function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
    canvas_width = canvas.width;
    canvas_height = canvas.height;

    // canvas.addEventListener("mousemove", function (e) {
    //     findxy('move', e)
    // }, false);
    // canvas.addEventListener("mousedown", function (e) {
    //     findxy('down', e)
    // }, false);
    // canvas.addEventListener("mouseup", function (e) {
    //     findxy('up', e)
    // }, false);
    // canvas.addEventListener("mouseout", function (e) {
    //     findxy('out', e)
    // }, false);
    // resizeTo(canvas, 4)
}


function generate_fractal_noise(width, height) {

	Permutation = MakePermutation();
	const pixels = new Array(height).fill(0).map(() => new Array(width).fill(0));

	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			// Noise2D generally returns a value approximately in the range [-1.0, 1.0]
			n = Noise2D(x*0.01, y*0.01);
			
			// Transform the range to [0.0, 1.0], supposing that the range of Noise2D is [-1.0, 1.0]
			n += 1.0;
			n /= 2.0;
			
			c = Math.round(255*n);
			pixels[y][x] = c;
		}
	}

	return pixels;
}

function draw_noise_on_canvas() {
	pixels = generate_fractal_noise(canvas_width, canvas_height);
	pixels = apply_threshold(pixels)
	draw_matrix_on_canvas(pixels);
}

function draw_matrix_on_canvas(matrix) {
	ctx = canvas.getContext("2d")
	ctx.clearRect(0, 0, canvas_width, canvas_height)

	// pixels = generate_fractal_noise(width, height)

	for (var y = 0; y < canvas_height; y++) {
		for (var x = 0; x < canvas_width; x++) {
			// console.log(pixels[y][x])
			var color = "rgba(0, 0, 0," + (1 - (matrix[y][x] / 255)) + ")"
			ctx.fillStyle = color
			ctx.fillRect(x, y, 1, 1) 
		}
	}
}

function apply_threshold(matrix) {
	threshold_val = 100

	for (var y = 0; y < matrix.length; y++) {
		for (var x = 0; x < matrix[0].length; x++) {
			matrix[y][x] -= threshold_val;
			matrix[y][x] = Math.max(0, matrix[y][x])
		}
	}

	return matrix;
}

