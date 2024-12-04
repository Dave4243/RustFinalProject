use crate::model::{Network, Layer};


impl NetworkDim {
    pub fn draw(T: Drawable) {
        if self == NetworkDim::Invalid {panic!("Egg")}
    }
}

pub trait Drawable {
    fn check(self: &Self) -> NetworkDim;

}
impl Drawable for Network {
    fn check(self:&Self) -> NetworkDim {
        match self.input_layer().size {
            2 => NetworkDim::Network2D,
            3 => NetworkDim::Network3D,
            _ => NetworkDim::Invalid,
        }
    }
}

impl Network_2D for Network {
    fn new(weights: Vec<Vec<f64>>) -> Self
    {

    }

}
// impl Network for Network_2D {}
