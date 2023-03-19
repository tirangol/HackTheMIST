// import { createNoise2D } from './node_modules/simplex-noise/dist/cjs/simplex-noise.js';
// import {createNoise2D} from 'simplex-noise'
// import {createNoise2D} from "https://cdn.skypack.dev/simplex-noise@4.0.0";

// const { createNoise2D } = require("simplex-noise");
// import { __esModule } from 'simplex-noise/dist/cjs/simplex-noise';
// import {simplex2D} from './simplex.js'


// createNoise2D
// let gen = createNoise2D();
// import the noise functions you need
// const { createNoise2D } = require('simplex-noise');

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
var canvas2
var ctx;
var width;
var height;
var canvas_width;
var canvas_height;
var canvas_state;
var elevation_matrix;

var scale_factor;

var prevX = 0
var currX = 0
var prevY = 0
var currY = 0
var dot_flag = false
var flag = false
var threshold_mat;

window.onload = function initialize_things() {
    canvas = document.getElementById('can');
	canvas2 = document.getElementById('can2')
	canvas.oncontextmenu = function (e) {
		e.preventDefault();
	};
    ctx = canvas.getContext("2d");

    canvas_width = canvas.width;
    canvas_height = canvas.height;

	width = 360
	height = 180

	scale_factor = canvas_width / width;

	canvas_state = new Array(height).fill(0).map(() => new Array(width).fill(0));
	elevation_matrix = new Array(width).fill(0).map(() => new Array(width).fill(0));
	threshold_mat = new Array(canvas_height).fill(0).map(() => new Array(width).fill(0));
	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			threshold_mat[y][x]	= 140
		}
	}

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
    // resizeTo(canvas, 4)
}


function set_random_elevation() {

	Permutation = MakePermutation();
	// const pixels = new Array(height).fill(0).map(() => new Array(width).fill(0));

	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			// Noise2D generally returns a value approximately in the range [-1.0, 1.0]
			var n = 0.5 * Noise2D(x*0.015, y*0.015);
			n += 0.25 * Noise2D(x*0.03, y*0.03);
			n += 0.125 * Noise2D(x*0.08, y*0.08);
			// n += 0.05 * Noise2D(x*0.08, y*0.08);
			// var n = simplex2D(x*0.01, y*0.01);
			
			// Transform the range to [0.0, 1.0], supposing that the range of Noise2D is [-1.0, 1.0]
			n += 1.0;
			n /= 2.0;
			
			var c = Math.round(255*n);
			elevation_matrix[y][x] = c;
		}
	}
}




window.draw_noise_on_canvas=()=>{
	set_random_elevation()
	// console.log(elevation_matrix)
	// var pixels = generate_fractal_noise(canvas_width, canvas_height);
	apply_threshold()
	draw_matrix_on_canvas(canvas_state);
}

function draw_matrix_on_canvas(matrix) {
	ctx = canvas.getContext("2d")
	ctx.fillStyle = "black"
	ctx.fillRect(0, 0, canvas_width, canvas_height)

	// pixels = generate_fractal_noise(width, height)

	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			// console.log(pixels[y][x])
			var opacity = matrix[y][x] / 40
			// var opacity;
			// if (matrix[y][x] > 0) {
			// 	opacity = 255
			// } else {
			// 	opacity = 0
			// }
			var color = "rgba(255, 255, 255," + opacity + ")"
			ctx.fillStyle = color
			// console.log(2*x, 2*y)
			ctx.fillRect(scale_factor * x, scale_factor * y, scale_factor, scale_factor) 
		}
	}

	// ctx.fillStyle = "green"
	// ctx.fillRect(300, 180, 20, 20)
}

function apply_threshold() {
	// var sea_level = 140

	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			// canvas_state[y][x] -= threshold_mat[y][x];
			canvas_state[y][x] = Math.max(0, elevation_matrix[y][x] - threshold_mat[y][x])
		}
	}

	// return matrix;
}

function in_bounds(x, y) {
	if (x < 0 || x >= width) return false
	if (y < 0 || y >= height) return false
	return true
}


function draw(sign) {
    // ctx.beginPath();
    // ctx.moveTo(prevX, prevY);
    // ctx.lineTo(currX, currY);
    // ctx.strokeStyle = "white";
    // ctx.lineWidth = 10;
    // ctx.stroke();
    // ctx.closePath();


	var stroke = 15;
	var scaled_x = Math.floor(currX / scale_factor)
	var scaled_y = Math.floor(currY / scale_factor)

	var min_x = scaled_x - stroke;
	var max_x = scaled_x + stroke;

	var min_y = scaled_y - stroke;
	var max_y = scaled_y + stroke;

	// console.log(min_x, max_x)
	// console.log(min_y, max_y)

	// console.log("poo")

	for (var y = min_y; y <= max_y; y++) {
		for (var x = min_x; x <= max_x; x++) {
			if (!in_bounds(x,y)) 
				continue
			var dist = ((y - scaled_y)**2 + (x - scaled_x)**2) / 30	
			var delta = 8 - dist
			// delta *= 0.5
			delta = sign * Math.max(0, delta)
			// console.log(delta)

			// console.log(delta)
			elevation_matrix[y][x] += delta
			elevation_matrix[y][x] = Math.max(0, elevation_matrix[y][x])
			elevation_matrix[y][x] = Math.min(255, elevation_matrix[y][x])
		}
	}
	apply_threshold()
	draw_matrix_on_canvas(canvas_state)
	// draw_matrix_on_canvas(elevation_matrix)

}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
			if (e.button == 0) {
				draw(1)
			} else if (e.button == 2) {
				draw(-1)
			}
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }

	// console.log(currX, currY)
    // if (res == 'move') {
    //     if (flag) {
    //         prevX = currX;
    //         prevY = currY;
    //         currX = e.clientX - canvas.offsetLeft;
    //         currY = e.clientY - canvas.offsetTop;
    //         draw();
    //     }
    // }
}

function normalize() {
	// console.log(canvas_state)
	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			canvas_state[y][x] /= 115; 
			// console.log(canvas_state[y][x])
			// console.log(y, x)
		}
	}
	return canvas_state
}

async function obtain_colorized() {

	var normalized_elevations = normalize()

	const path = `http://localhost:5000/predict`;
    const res = await fetch(path, {
		method: 'POST',
		headers: {
			'Accept': 'application/json',
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(normalized_elevations)
	});

    const response = await res.json();
	// console.log(response)

	var ctx2 = canvas2.getContext("2d");


	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			var r = response[y][x][0]
			var g = response[y][x][1]
			var b = response[y][x][2]

			var color = `rgba(${r}, ${g}, ${b}, 1)`
			// console.log(color)
			ctx2.fillStyle = color
			// console.log(2*x, 2*y)
			ctx2.fillRect(scale_factor * x, scale_factor * y, scale_factor, scale_factor)
		}
	}
}




// module.exports = {
// 	draw_noise_on_canvas,
// };

// console.log(simplex2D(0, 100))