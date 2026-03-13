use std::fmt::Debug;

use futures_util::StreamExt;
use langchain_core::stores::BaseStore;

pub trait BaseStoreHarness {
    type Store: BaseStore<Self::Value>;
    type Value: Clone + Debug + PartialEq + Send + Sync + 'static;

    fn store(&self) -> Self::Store;

    fn three_values(&self) -> (Self::Value, Self::Value, Self::Value);
}

pub struct BaseStoreSyncTests<H> {
    harness: H,
}

impl<H> BaseStoreSyncTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> BaseStoreSyncTests<H>
where
    H: BaseStoreHarness,
{
    pub fn run(&self) {
        let store = self.harness.store();
        assert_eq!(
            store.mget(&["foo".to_owned(), "bar".to_owned()]),
            vec![None, None]
        );

        let (foo, bar, baz) = self.harness.three_values();
        store.mset(vec![
            ("foo".to_owned(), foo.clone()),
            ("bar".to_owned(), bar.clone()),
            ("baz".to_owned(), baz.clone()),
        ]);
        assert_eq!(
            store.mget(&["foo".to_owned(), "bar".to_owned(), "baz".to_owned()]),
            vec![Some(foo.clone()), Some(bar.clone()), Some(baz.clone())]
        );

        store.mdelete(&["foo".to_owned(), "baz".to_owned()]);
        assert_eq!(
            store.mget(&["foo".to_owned(), "bar".to_owned(), "baz".to_owned()]),
            vec![None, Some(bar)]
                .into_iter()
                .chain([None])
                .collect::<Vec<_>>()
        );
        assert_eq!(store.yield_keys(Some("b")).collect::<Vec<_>>(), vec!["bar"]);
    }
}

pub trait BaseStoreAsyncHarness: BaseStoreHarness {}

pub struct BaseStoreAsyncTests<H> {
    harness: H,
}

impl<H> BaseStoreAsyncTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> BaseStoreAsyncTests<H>
where
    H: BaseStoreAsyncHarness,
{
    pub async fn run(&self) {
        let store = self.harness.store();
        assert_eq!(
            store.amget(vec!["foo".to_owned(), "bar".to_owned()]).await,
            vec![None, None]
        );

        let (foo, bar, baz) = self.harness.three_values();
        store
            .amset(vec![
                ("foo".to_owned(), foo.clone()),
                ("bar".to_owned(), bar.clone()),
                ("baz".to_owned(), baz.clone()),
            ])
            .await;
        assert_eq!(
            store
                .amget(vec!["foo".to_owned(), "bar".to_owned(), "baz".to_owned()])
                .await,
            vec![Some(foo.clone()), Some(bar.clone()), Some(baz.clone())]
        );

        store
            .amdelete(vec!["foo".to_owned(), "baz".to_owned()])
            .await;
        assert_eq!(
            store
                .amget(vec!["foo".to_owned(), "bar".to_owned(), "baz".to_owned()])
                .await,
            vec![None, Some(bar), None]
        );
        assert_eq!(
            store.ayield_keys(Some("b")).collect::<Vec<_>>().await,
            vec!["bar"]
        );
    }
}
