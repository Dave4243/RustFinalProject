use std::vec::Vec;
use rand::Rng;
use std::fs::File;
use std::io::{self, BufReader, Read};
#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Relu,
    Softmax,
}
impl ActivationFunction {
    pub fn calculate(self, input: f64) -> f64 {
        match self {
            Self::Relu => input.max(0.0),
            Self::Softmax => input, // needs input vector, so this is handled in function
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    weights: Vec<Vec<f64>>,

    biases: Vec<f64>,

    input : Vec<f64>,

    z_values : Vec<f64>,

    activation_function: ActivationFunction,
}

impl Layer { 
    // initalizes random weights and biases, 
    fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        
        // He initalization
        let limit = (2.0 / input_size as f64).sqrt(); 
        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..input_size)
                .map(|_| rng.gen_range(-limit..limit))
                .collect())
            .collect();

        let biases: Vec<f64> = (0..output_size).map(|_| rng.gen_range(-0.01..0.01)).collect();

        Self {
            weights,
            biases,
            input : vec![0.0; input_size],
            z_values : vec![],
            activation_function : activation,
        }
    }

    // function to initiate a forward pass through this layer of neurons
    pub fn calculate(&mut self, prev_layer_out: &Vec<f64>) -> Vec<f64> {
        self.input = prev_layer_out.clone();
        let mut result: Vec<f64> = vec![0.0; self.weights.len()]; 

        // Col is the index of the layer vector
        for (col, weights_row) in self.weights.iter().enumerate() {
            let mut sum: f64 = weights_row.iter()
                .zip(prev_layer_out)
                .map(|(x, y)| x*y)
                .sum();
            sum += self.biases[col];
            result[col] = sum;
        }
        // Store the pre-activation values
        self.z_values = result.clone();
        
        match self.activation_function {
            ActivationFunction::Softmax => {
                // Find the maximum value for numerical stability
                let max = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = result.iter().map(|&x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                result = exps.iter().map(|&x| x / sum).collect();
            },
            _ => result.iter_mut().for_each(|x| *x = self.activation_function.calculate(*x)),
        }
        // Store results and return
        return result;
    }

    /// Backprop
    fn backpropagate(&mut self, output_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let dLdz : Vec<f64>;
        match self.activation_function {
            ActivationFunction::Relu => {
                dLdz = output_gradient.iter()
                .zip(self.z_values.iter())
                .map(|(&og, &iv)| if iv > 0.0 { og } else { 0.0 })
                .collect();
            },
            ActivationFunction::Softmax => dLdz = output_gradient,
            _ => return Vec::new(),
        }
        // computes gradient with respect to input (output of prev layer)
        let mut dLdx = vec![0.0; self.input.len()];
        for j in 0..self.input.len() {
            for i in 0..self.weights.len() {
                dLdx[j] += dLdz[i] * self.weights[i][j];
            }
        }
        
        // update biases
        for i in 0..self.biases.len() {
            let dLdb = dLdz[i];
            self.biases[i] -= learning_rate * dLdb;
        }
        // update weights
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                let dLdW = dLdz[i] * self.input[j];
                self.weights[i][j] -= learning_rate * dLdW;
            }
        }
        // pass this gradient back
        return dLdx;
    }
}

#[derive(Clone)]
pub struct Network {
    layers: Vec<Layer>,
    learning_rate : f64,
}

impl Network{
    pub fn new() -> Self {
        let mut network = Network {
            layers : vec![],
            learning_rate : 0.05,
        };
        // single layer neural network
        let first_layer = Layer::new
        (784, 512, ActivationFunction::Relu);
        let output_layer = Layer::new
        (512, 10, ActivationFunction::Softmax);
        network.layers.push(first_layer);
        network.layers.push(output_layer);
        return network;
    }

    pub fn calculate(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut curr_input: Vec<f64> = input.clone();

        for curr_layer in self.layers.iter_mut() {
            curr_input = curr_layer.calculate(&curr_input);
        }
        return curr_input;
    }
    
    // actual_classification is the vector of 10 values containing 0s 
    // and 1 for the actual number the model was supposed to predict
    pub fn backpropagate(&mut self, output: &Vec<f64>, actual_classification: &Vec<f64>) -> f64 {
        let fake_zero = 1e-10;
        let loss: f64 = output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| -(target * (output.max(fake_zero).ln())))
            .sum::<f64>();        

        // Compute the gradient of the loss (cross-entropy)
        let output_gradient: Vec<f64> = output
            .iter()
            .zip(actual_classification)
            .map(|(&output, &target)| output - target) // Gradient for softmax with cross-entropy
            .collect();

        // Perform backpropagation for each layer
        let mut gradient = output_gradient; // Initialize with the gradient of the output layer
        for i in (0..self.layers.len()).rev() {
            gradient = self.layers[i].backpropagate(gradient, self.learning_rate);
        }
        return loss;
    }

    // REAL TRAIN
    pub fn train(&mut self, images_filename: &str, labels_filename: &str, iterations: usize) -> io::Result<()> {
        let mut image_reader = ImageReader::new(images_filename)?;
        let mut label_reader = LabelReader::new(labels_filename)?;

        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut average_loss = 0.0;
        let training_size = 2000;
        for i in 0..training_size {
            let image = image_reader.read_next_image().unwrap().unwrap();
            images.push(image);
            let label = label_reader.read_next_label().unwrap().unwrap();
            labels.push(label);
        }
        for iteration in 0..iterations {
            let image = &images[iteration % training_size];
            let label = labels[iteration % training_size];
            let prediction = self.calculate(&image);
            let mut actual_classification = vec![0.0; 10];
            actual_classification[label as usize] = 1.0;


            let loss = self.backpropagate(&prediction, &actual_classification);
            average_loss += loss;
            if iteration % 50 == 0 {
                average_loss /= 50.0;
                println!("Iteration {} completed, Average Loss over last 50 iterations = {}", iteration, average_loss);
                if iteration % 500 == 0 {
                    self.learning_rate *= 0.95;
                    self.learning_rate = self.learning_rate.max(0.00001);
                }
                average_loss = 0.0;
            }
            if iteration % 1000 == 0 {
                let mut rng = rand::thread_rng();
                for i in 0..1000 {
                    let j = rng.gen_range(i..1000);
                    images.swap(i, j);  
                    labels.swap(i, j);
                }
            }
        }
    
        Ok(())
    }
}
struct ImageReader {
    reader: BufReader<File>,
    image_size: usize,
}

impl ImageReader {
    fn new(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        // Read the magic number
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        let magic_number = u32::from_be_bytes(buffer);
        if magic_number != 2051 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number"));
        }
        // Read the number of images (not used directly here)
        reader.read_exact(&mut buffer)?;

        // Read the number of rows
        reader.read_exact(&mut buffer)?;
        let num_rows = u32::from_be_bytes(buffer) as usize;

        // Read the number of columns
        reader.read_exact(&mut buffer)?;
        let num_cols = u32::from_be_bytes(buffer) as usize;
        // Verify dimensions are 28x28
        if num_rows != 28 || num_cols != 28 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Expected 28x28 images"));
        }
        let image_size = num_rows * num_cols;

        Ok(ImageReader { reader, image_size })
    }

    fn read_next_image(&mut self) -> io::Result<Option<Vec<f64>>> {
        let mut buffer = vec![0u8; self.image_size];
        let bytes_read = self.reader.read(&mut buffer)?;

        if bytes_read == 0 {
            return Ok(None); // End of file
        }

        let image: Vec<f64> = buffer.iter().map(|&x| x as f64 / 255.0).collect();
        Ok(Some(image))
    }
}

struct LabelReader {
    reader: BufReader<File>,
}

impl LabelReader {
    fn new(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        // Read the header
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        let magic_number = u32::from_be_bytes(buffer);
        if magic_number != 2049 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number"));
        }

        // Read the number of labels
        reader.read_exact(&mut buffer)?;

        Ok(LabelReader  {reader})
    }

    fn read_next_label(&mut self) -> io::Result<Option<u8>> {
        let mut buffer = [0u8; 1];
        let bytes_read = self.reader.read(&mut buffer)?;

        if bytes_read == 0 {
            return Ok(None); // End of file
        }

        Ok(Some(buffer[0]))
    }
}