use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};

use winit::{event, keyboard};

#[derive(Debug, PartialEq)]
pub enum MouseOrKey {
    Mouse(event::MouseButton),
    Key(keyboard::PhysicalKey),
}

#[derive(Debug)]
pub struct KeyBind {
    mok: MouseOrKey,
    label: String,
}

impl KeyBind {
    pub fn new(mok: MouseOrKey) -> Self {
        let label: String = match mok {
            MouseOrKey::Mouse(b) => serde_json::to_string(&b).unwrap(),
            MouseOrKey::Key(p) => match p {
                keyboard::PhysicalKey::Code(x) => serde_json::to_string(&x).unwrap(),
                keyboard::PhysicalKey::Unidentified(x) => match x {
                    keyboard::NativeKeyCode::Android(x) => x.to_string(),
                    keyboard::NativeKeyCode::MacOS(x) => x.to_string(),
                    keyboard::NativeKeyCode::Windows(x) => x.to_string(),
                    keyboard::NativeKeyCode::Xkb(x) => x.to_string(),
                    _ => "".to_string(),
                },
            },
        };
        Self { mok, label }
    }
}

impl Display for KeyBind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

pub struct Controller {
    bind_actions: HashMap<usize, KeyBind>,
    hold_keys: HashSet<usize>,
    mouse_data: (u64, cgmath::Vector2<f64>),
    old_mouse_data: (u64, cgmath::Vector2<f64>),
}

impl Controller {
    pub fn new(window_width: f64, window_height: f64) -> Self {
        Controller {
            bind_actions: HashMap::new(),
            hold_keys: HashSet::new(),
            mouse_data: (0, cgmath::vec2(0.0, 0.0)),
            old_mouse_data: (0, cgmath::vec2(window_width / 2.0, window_height / 2.0)),
        }
    }

    pub fn register_bind_action(&mut self, action: usize, key_bind: KeyBind) {
        self.bind_actions.insert(action, key_bind);
    }

    pub fn handle_event(
        &mut self,
        event: event::WindowEvent,
        frames: u64,
        window_origin: &cgmath::Vector2<f64>,
    ) {
        match event {
            event::WindowEvent::KeyboardInput {
                device_id: _,
                event: key_event,
                is_synthetic: _,
            } => {
                let keybind = self.retrieve(&MouseOrKey::Key(key_event.physical_key));
                match keybind {
                    None => {}
                    Some(action) => {
                        if key_event.state.is_pressed() {
                            self.hold_keys.insert(action);
                        } else {
                            self.hold_keys.remove(&action);
                        }
                    }
                }
            }
            event::WindowEvent::ModifiersChanged(_) => {}
            event::WindowEvent::CursorMoved { .. } => {}
            event::WindowEvent::CursorEntered { .. } => {}
            event::WindowEvent::CursorLeft { .. } => {}
            event::WindowEvent::MouseWheel { .. } => {}
            event::WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                let keybind = self.retrieve(&MouseOrKey::Mouse(button));
                match keybind {
                    None => {}
                    Some(action) => {
                        if state.is_pressed() {
                            self.hold_keys.insert(action);
                        } else {
                            self.hold_keys.remove(&action);
                        }
                    }
                }
            }
            event::WindowEvent::AxisMotion {
                device_id: _,
                axis,
                value,
            } => {
                if self.mouse_data.1.x > 0.02
                    && self.mouse_data.1.y > 0.02
                    && frames > self.mouse_data.0
                {
                    self.old_mouse_data = self.mouse_data.clone()
                }
                match axis {
                    0 => self.mouse_data = (frames, cgmath::vec2(value, self.mouse_data.1.y)),
                    1 => self.mouse_data = (frames, cgmath::vec2(self.mouse_data.1.x, value)),
                    _ => panic!("Unknown axis {axis}"),
                };
                // println!(
                //     "axis={axis:?} frame={frames:?} new={a:?} old={b:?}",
                //     a = self.mouse_data,
                //     b = self.old_mouse_data
                // )
            }
            _ => (), // println!("{:?}", event),
        }
    }

    pub fn mouse_direction(&self, frames: u64) -> cgmath::Vector2<f64> {
        if self.mouse_data.1.x > 0.02
            && self.mouse_data.1.y > 0.02
            && self.mouse_data.0 > frames - 10
        // TODO get number of frames in last second ?
        {
            cgmath::vec2(
                self.mouse_data.1.x - self.old_mouse_data.1.x,
                self.mouse_data.1.y - self.old_mouse_data.1.y,
            )
        } else {
            cgmath::vec2(0.0, 0.0)
        }
    }

    pub fn mouse_velocity(&self, window_width: f64, window_height: f64) -> cgmath::Vector2<f64> {
        if self.mouse_data.1.x > 0.02 && self.mouse_data.1.y > 0.02 {
            cgmath::vec2(
                self.mouse_data.1.x - self.old_mouse_data.1.x,
                self.mouse_data.1.y - self.old_mouse_data.1.y,
            )
        } else {
            cgmath::vec2(
                self.old_mouse_data.1.x - window_width / 2.0,
                self.old_mouse_data.1.y - window_height / 2.0,
            )
        }
    }

    pub fn is_hold(&self, action: usize) -> bool {
        self.hold_keys.contains(&action)
    }

    fn retrieve(&self, mok: &MouseOrKey) -> Option<usize> {
        self.bind_actions.iter().find_map(
            |(k, v)| {
                if &v.mok == mok { Some(k.clone()) } else { None }
            },
        )
    }
}
