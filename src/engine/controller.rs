use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};

use winit::{event, keyboard};

#[derive(Debug)]
pub struct KeyBind {
    key: keyboard::PhysicalKey,
    label: String,
}

impl KeyBind {
    pub fn new(key: keyboard::PhysicalKey) -> Self {
        let label: String = match key {
            keyboard::PhysicalKey::Code(x) => serde_json::to_string(&x).unwrap(),
            keyboard::PhysicalKey::Unidentified(x) => match x {
                keyboard::NativeKeyCode::Android(x) => x.to_string(),
                keyboard::NativeKeyCode::MacOS(x) => x.to_string(),
                keyboard::NativeKeyCode::Windows(x) => x.to_string(),
                keyboard::NativeKeyCode::Xkb(x) => x.to_string(),
                _ => "".to_string(),
            },
        };
        Self { key, label }
    }

    #[allow(dead_code)]
    pub fn eq(&self, key: &keyboard::PhysicalKey) -> bool {
        &self.key == key
    }
}

impl Display for KeyBind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

// impl Hash for KeyBind {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.key.hash(state)
//     }
// }
//
// impl PartialEq<Self> for KeyBind {
//     fn eq(&self, other: &Self) -> bool {
//         self.key == other.key
//     }
// }

pub struct Controller {
    bind_actions: HashMap<usize, KeyBind>,
    hold_keys: HashSet<usize>,
}

impl Controller {
    pub fn new() -> Self {
        Controller {
            bind_actions: HashMap::new(),
            hold_keys: HashSet::new(),
        }
    }
    pub fn register_bind_action(&mut self, action: usize, key_bind: KeyBind) {
        self.bind_actions.insert(action, key_bind);
    }

    pub fn handle_event(&mut self, event: event::WindowEvent) {
        match event {
            event::WindowEvent::KeyboardInput {
                device_id: _,
                event: key_event,
                is_synthetic: _,
            } => {
                let keybind = self.retrieve(&key_event.physical_key);
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
            event::WindowEvent::MouseInput { .. } => {}
            event::WindowEvent::AxisMotion { .. } => {}
            _ => (), // println!("{:?}", event),
        }
    }

    pub fn is_hold(&self, action: usize) -> bool {
        self.hold_keys.contains(&action)
    }

    fn retrieve(&self, key: &keyboard::PhysicalKey) -> Option<usize> {
        self.bind_actions.iter().find_map(
            |(k, v)| {
                if &v.key == key { Some(k.clone()) } else { None }
            },
        )
    }
}
