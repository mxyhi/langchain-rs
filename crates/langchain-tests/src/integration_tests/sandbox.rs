use crate::base::BaseStandardTests;

pub trait SandboxIntegrationHarness {
    type Sandbox;

    fn sandbox(&self) -> Self::Sandbox;

    fn has_sync(&self) -> bool {
        true
    }

    fn has_async(&self) -> bool {
        true
    }
}

pub struct SandboxIntegrationTests<H> {
    harness: H,
    base: BaseStandardTests,
}

impl<H> SandboxIntegrationTests<H> {
    pub fn new(harness: H) -> Self {
        Self {
            harness,
            base: BaseStandardTests::new("sandbox"),
        }
    }

    pub const fn base(&self) -> &BaseStandardTests {
        &self.base
    }
}

impl<H> SandboxIntegrationTests<H>
where
    H: SandboxIntegrationHarness,
{
    pub fn supports_sync(&self) -> bool {
        self.harness.has_sync()
    }

    pub fn supports_async(&self) -> bool {
        self.harness.has_async()
    }

    pub fn sandbox(&self) -> H::Sandbox {
        self.harness.sandbox()
    }
}
