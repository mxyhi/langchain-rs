//! Legacy/classic compatibility surface.
//!
//! This crate mirrors the role of Python `langchain_classic`: it is the landing
//! zone for APIs that belong to the legacy/classic package rather than the new
//! facade crates. The implementation is intentionally minimal for now, but the
//! package boundary is real and ready to absorb classic-only modules later.

/// Marker type for the classic/legacy package boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassicPackage;

impl ClassicPackage {
    /// Canonical crate name used in the Rust workspace.
    pub const fn package_name(self) -> &'static str {
        "langchain-classic"
    }

    /// Short explanation of why this crate exists.
    pub const fn purpose(self) -> &'static str {
        "Legacy compatibility surface corresponding to Python langchain_classic."
    }
}
