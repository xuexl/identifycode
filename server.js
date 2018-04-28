// https://github.com/karpathy/convnetjs

const http = require('http');
const querystring = require('querystring');
const path = require('path');
const fs = require('fs');
const conv = require('convnetjs');

const hostname = '127.0.0.1';
const port = 3001;
const root = __dirname;

const server = http.createServer((req, res) => {
	con();
	index(res);
});

server.listen(port, hostname, () => {

	console.log(`Server running at http://${hostname}:${port}/`);
});

function index(res){	
	fs.readFile('index.html', (err, content)=>{
		res.statusCode = 200;
		res.setHeader('Content-Type', 'text/html');	
		res.write(content);
		res.end();
	});	
}
//
function con(){
	con1();
	
}
//
function con1(){
	var layer_defs = [];
	// input layer declares size of input. here: 2-D data
	// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
	// then the first two dimensions (sx, sy) will always be kept at size 1
	layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
	// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
	layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
	// declare the linear classifier on top of the previous hidden layer
	layer_defs.push({type:'softmax', num_classes:10});

	var net = new convnetjs.Net();
	net.makeLayers(layer_defs);

	// forward a random data point through the network
	var x = new convnetjs.Vol([0.3, -0.5]);
	var prob = net.forward(x);

	// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
	console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

	var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
	trainer.train(x, 0); // train the network, specifying that x is class zero

	var prob2 = net.forward(x);
	console.log('probability that x is class 0: ' + prob2.w[0]);

}




