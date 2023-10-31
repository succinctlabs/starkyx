use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

use log::{set_max_level, LevelFilter};

static INIT: Once = Once::new();

pub fn setup_logger() {
    INIT.call_once(|| {
        env_logger::Builder::from_default_env()
            .format_timestamp(None)
            .filter_level(LevelFilter::Trace)
            .init();
    });
}

static ORIGINAL_LEVEL: AtomicUsize = AtomicUsize::new(LevelFilter::Info as usize);

pub fn disable_logging() {
    let current_level = log::max_level() as usize;
    ORIGINAL_LEVEL.store(current_level, Ordering::SeqCst);
    set_max_level(LevelFilter::Off);
}

pub fn enable_logging() {
    let original_level = ORIGINAL_LEVEL.load(Ordering::SeqCst);
    set_max_level(unsafe { std::mem::transmute(original_level) });
}
