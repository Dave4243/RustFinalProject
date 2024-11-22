// struct Uniforms {
//     bounds: vec4<u32>
// };

// struct input_data {
//     weights: array<u32>,
//     prev_layer: array<u32>,
// };
const E: f32 = 2.71828182845904523536028747135266250;


struct OutputData {
    layer_output: array<u32>
};

// enum ActivationFunctions {
//     Sigmoid,
//     Tanh,
//     Relu,
//     None,
// }

struct LayerData {
    size:u32,
    activation: u32 // enum ActivationFunctions
}

struct BufferData {
    // prev_layer_size: u32,
    weights_dims: vec2<u32>,
    curr_layer_index: u32,
    // layer_sizes: array<u32>,
    // output_row_max_size: u32,
    // Contains size of every layer, including input layer at index 0
    // This means it is not one to one with the networks layers vector!
    output_layer_sizes_and_activations: array<LayerData>
}


// @group(0) @binding(0) var<uniform> uniforms: Uniforms;
// @group(0) @binding(1) var<storage, read> prev_layer_buffer: array<f32>;
@group(0) @binding(2) var<storage, read> weights_buffer: array<f32>;
@group(0) @binding(3) var<storage, read> buffer_data_buffer: BufferData;

// Each row represents each layer output, including the initial values of the input layer!
@group(0) @binding(0) var<storage, read_write> write_buffer: array<f32>;

fn is_outside_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
    return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y) || coord.z >= u32(bounds.z);
}

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (is_outside_bounds(global_id, uniforms.bounds.xyz)) {
    // if (is_outside_bounds(global_id, vec3<u32>(buffer_data_buffer.weights_dims.xy,1))) {
    let curr_layer_size = buffer_data_buffer.output_layer_sizes_and_activations[buffer_data_buffer.curr_layer_index].size;
    if (is_outside_bounds(global_id, vec3<u32>(curr_layer_size, 1,1))) {
        return;
    }

    // if (buffer_data_buffer.curr_layer_index >= arrayLength(&buffer_data_buffer.layer_sizes)) {
    //     return;
    // }

    var sum: f32 = 0f;  

    for (var i:u32=0; i<buffer_data_buffer.weights_dims.x; i++) {
        sum += read_out(buffer_data_buffer.curr_layer_index-1, i) * read_weights(global_id.y, i);
    }

    // Run through activation function
    var cooked_sum: f32 = -0.0;
    switch(buffer_data_buffer.output_layer_sizes_and_activations[buffer_data_buffer.curr_layer_index].size) {
        case 0u: { // Sigmoid
            let B:f32 = 1.0;
            cooked_sum = 1.0/(pow(E, -sum * B));
            // cooked_sum = -0.0;
        }
        case 1u: { // Tanh
            // cooked_sum = -0.0;
            cooked_sum = (pow(E,sum)-pow(E, -sum)) / (pow(E, sum) + pow(E, -sum));
        }
        case 2u: { // Relu
            // cooked_sum = -0.0;
        }
        case 3u: { // None
            // cooked_sum = sum;
        }
        default: {
            cooked_sum = -0.0;
        }
    }

    write_out(buffer_data_buffer.curr_layer_index, global_id.x, cooked_sum);

}

fn read_out(row: u32, col: u32) -> f32 {
    // if (col > buffer_data_buffer.weights_dims.x || row > buffer_data_buffer.weights_dims.y) {return -0.0;}
    // size 

    var row_offset:u32 = 0;
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.output_layer_sizes_and_activations[i].size;
    }

    if (col > buffer_data_buffer.output_layer_sizes_and_activations[row].size) {return -0.0;}

    return write_buffer[col + row_offset];
    // return write_buffer[col + row * buffer_data_buffer.weights_dims.x];
}

fn write_out(row: u32, col: u32, data: f32) {
    // This is also wrong
    var row_offset:u32 = 0;
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.output_layer_sizes_and_activations[i].size;
    }

    write_buffer[col + row_offset] = data;
}

fn read_weights(row: u32, col: u32) -> f32 {
    // buffer_data_buffer.
    if (col > buffer_data_buffer.weights_dims.x || row > buffer_data_buffer.weights_dims.y) {return -0.0;}
    return weights_buffer[col + row * buffer_data_buffer.weights_dims.x];
}