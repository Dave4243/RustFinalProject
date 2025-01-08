// struct Uniforms {
//     bounds: vec4<u32>
// };

// struct input_data {
//     weights: array<u32>,
//     prev_layer: array<u32>,
// };
// const E: f32 = 2.71828182845904523536028747135266250;


// struct OutputLayerData {
//     z_value: u32,
//     layer_output: u32
// };

struct OutputData {
    z_value: f32,
    layer_output: f32
};

// enum ActivationFunctions {
//     Sigmoid,
//     Tanh,
//     Relu,
//     None,
// }

struct LayerData {
    size:u32,
    activation: u32, // enum ActivationFunctions
    layer_offset: u32 // Precomputed offsets telling where in bias buffer to find proper biases
}

struct BufferData {
    // prev_layer_size: u32,
    weights_dims: vec2<u32>,
    curr_layer_index: u32,
    // layer_sizes: array<u32>,
    // output_row_max_size: u32,
    // Contains size of every layer, including input layer at index 0
    // This means it is not one to one with the networks layers vector!
    layer_datas: array<LayerData>
}


// @group(0) @binding(0) var<uniform> uniforms: Uniforms;
// @group(0) @binding(1) var<storage, read> prev_layer_buffer: array<f32>;
@group(0) @binding(2) var<storage, read> weights_buffer: array<f32>;
@group(0) @binding(3) var<storage, read> buffer_data_buffer: BufferData;
@group(0) @binding(4) var<storage, read> biases: array<f32>; // Finish implementation

// Each row represents each layer output, including the initial values of the input layer!
// @group(0) @binding(0) var<storage, read_write> write_buffer: array<f32>;
@group(0) @binding(0) var<storage, read_write> write_buffer: array<OutputData>;

fn is_outside_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
    return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y) || coord.z >= u32(bounds.z);
}

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (is_outside_bounds(global_id, uniforms.bounds.xyz)) {
    // if (is_outside_bounds(global_id, vec3<u32>(buffer_data_buffer.weights_dims.xy,1))) {
    let curr_layer_size = buffer_data_buffer.layer_datas[buffer_data_buffer.curr_layer_index].size;
    // DEBUG FROM HERE! WHEN WRITING OUT OF BOUNDS, z_value IS UPDATED!!
    if (is_outside_bounds(global_id, vec3<u32>(curr_layer_size, 1,1))) {
        return;
    }

    // if (buffer_data_buffer.curr_layer_index >= arrayLength(&buffer_data_buffer.layer_sizes)) {
    //     return;
    // }

    var sum: f32 = 0f;  

    for (var i:u32=0; i<buffer_data_buffer.weights_dims.x; i++) {
        sum += read_out_val(buffer_data_buffer.curr_layer_index-1, i) * read_weights(global_id.x, i);
    }
    
    // sum += read_bias(buffer_data_buffer.curr_layer_index, global_id.x); //biases[buffer_data_buffer.layer_datas[buffer_data_buffer.curr_layer_index][global_id.x] buffer_data_buffer.curr_layer_index][]
    // Add biases
    let bias_offset = buffer_data_buffer.layer_datas[buffer_data_buffer.curr_layer_index].layer_offset;
    sum += biases[bias_offset + global_id.x];
    
    // Write Z values
    write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, sum);
    // write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, f32(buffer_data_buffer.curr_layer_index));
    // write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, -15.0);
    // write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, f32(global_id.x));
    // write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, f32(curr_layer_size) + -25.0);
    // write_out_z(buffer_data_buffer.curr_layer_index, global_id.x, f32(curr_layer_size) + -25.0);

    // Run through activation function
    var cooked_sum: f32 = -0.0;
    // switch(buffer_data_buffer.layer_datas[buffer_data_buffer.curr_layer_index].size) {
    switch(buffer_data_buffer.layer_datas[buffer_data_buffer.curr_layer_index].activation) {
        case 0u: { // Sigmoid
            let B:f32 = 1.0;
            cooked_sum = 1.0/(exp(-sum * B));
            // cooked_sum = -0.0;
        }
        case 1u: { // Tanh
            // cooked_sum = -0.0;
            cooked_sum = (exp(sum)-exp(-sum)) / (exp(sum) + exp(-sum));
        }
        case 2u: { // Relu
            cooked_sum = max(0.0, sum);
        }
        case 3u: { // Softmax
            // AI generated code, take with grain of salt: (I could not be bothered to read the formula. I will return if issues arise.)
            var max_val: f32 = -99999.0;
            for (var i: u32 = 0; i < curr_layer_size; i++) {
                let val = read_out_z(buffer_data_buffer.curr_layer_index, i);
                if (val > max_val) {
                    max_val = val;
                }
            }

            var sum_exp: f32 = 0.0;
            for (var i: u32 = 0; i < curr_layer_size; i++) {
                sum_exp += exp(read_out_z(buffer_data_buffer.curr_layer_index, i) - max_val);
            }

            cooked_sum = exp(sum - max_val) / sum_exp;
        }
        case 4u: { // None
            cooked_sum = sum;
            // cooked_sum = -0.0;
        }
        default: {
            cooked_sum = -0.0;
        }
    }

    write_out_val(buffer_data_buffer.curr_layer_index, global_id.x, cooked_sum);
    // write_out_val(buffer_data_buffer.curr_layer_index, global_id.x, -10.0);

}

fn read_bias (layer: u32, index: u32) -> f32 {
    if (index < 0 || index > buffer_data_buffer.layer_datas[layer].size) {return -0.0;}

    return biases[buffer_data_buffer.layer_datas[layer].layer_offset + index];
    
    // return -0.0;
} 

// Use precomputed offsets instead of these time consuming helper functions
fn read_out_val(row: u32, col: u32) -> f32 {
    // if (col > buffer_data_buffer.weights_dims.x || row > buffer_data_buffer.weights_dims.y) {return -0.0;}
    // size 

    var row_offset:u32 = 0;
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.layer_datas[i].size;
    }

    if (col > buffer_data_buffer.layer_datas[row].size) {return -0.0;}

    return write_buffer[col + row_offset].layer_output;
    // return write_buffer[col + row * buffer_data_buffer.weights_dims.x];
}
fn read_out_z(row: u32, col: u32) -> f32 {
    // if (col > buffer_data_buffer.weights_dims.x || row > buffer_data_buffer.weights_dims.y) {return -0.0;}
    // size 

    var row_offset:u32 = 0;
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.layer_datas[i].size;
    }

    if (col > buffer_data_buffer.layer_datas[row].size) {return -0.0;}

    return write_buffer[col + row_offset].z_value;
    // return write_buffer[col + row * buffer_data_buffer.weights_dims.x];
}

fn write_out_val(row: u32, col: u32, data: f32) {
    var row_offset:u32 = 0;
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.layer_datas[i].size;
    }

    if (col < buffer_data_buffer.layer_datas[row].size) {
        write_buffer[col + row_offset].layer_output = data;
    }
}
fn write_out_z(row: u32, col: u32, data: f32) {
    // This is also wrong
    var row_offset:u32 = 0;
    // for (var i:u32=0; i<row; i++) {
    //     row_offset += buffer_data_buffer.layer_datas[i].size;
    // }
    for (var i:u32=0; i<row; i++) {
        row_offset += buffer_data_buffer.layer_datas[i].size;
    }

    if (col < buffer_data_buffer.layer_datas[row].size) {
        write_buffer[col + row_offset].layer_output = data;
    }

    write_buffer[col + row_offset].z_value = data;
    // write_buffer[col + row_offset].z_value = f32(row_offset);
}

fn read_weights(row: u32, col: u32) -> f32 {
    // buffer_data_buffer.
    if (col > buffer_data_buffer.weights_dims.x || row > buffer_data_buffer.weights_dims.y) {return -0.0;}
    return weights_buffer[col + row * buffer_data_buffer.weights_dims.x];
}