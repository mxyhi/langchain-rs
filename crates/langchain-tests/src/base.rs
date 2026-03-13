use std::collections::BTreeSet;

pub trait StandardTestSuite {
    fn running_tests(&self) -> &'static [&'static str];

    fn base_tests(&self) -> &'static [&'static str];

    fn overridden_tests(&self) -> &'static [&'static str] {
        &[]
    }

    fn xfail_overrides(&self) -> &'static [&'static str] {
        &[]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseStandardTests {
    suite_name: &'static str,
}

impl BaseStandardTests {
    pub const fn new(suite_name: &'static str) -> Self {
        Self { suite_name }
    }

    pub const fn suite_name(&self) -> &'static str {
        self.suite_name
    }

    pub fn assert_no_overrides<S>(&self, suite: &S) -> Result<(), String>
    where
        S: StandardTestSuite,
    {
        let running = suite
            .running_tests()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let base = suite.base_tests().iter().copied().collect::<BTreeSet<_>>();
        let deleted = base.difference(&running).copied().collect::<Vec<_>>();

        let xfail = suite
            .xfail_overrides()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let overridden = suite
            .overridden_tests()
            .iter()
            .copied()
            .filter(|test_name| !xfail.contains(test_name))
            .collect::<Vec<_>>();

        if deleted.is_empty() && overridden.is_empty() {
            return Ok(());
        }

        let mut errors = Vec::new();
        if !deleted.is_empty() {
            errors.push(format!("Standard tests deleted: {deleted:?}"));
        }
        if !overridden.is_empty() {
            errors.push(format!(
                "Standard tests overridden without xfail: {overridden:?}"
            ));
        }
        Err(errors.join("\n"))
    }
}
