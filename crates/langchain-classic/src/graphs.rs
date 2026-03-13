pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

macro_rules! define_graph_boundary {
    ($name:ident, $provider_name:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name {
            uri: String,
            remote_connected: bool,
        }

        impl $name {
            pub fn new(uri: impl Into<String>) -> Self {
                Self {
                    uri: uri.into(),
                    remote_connected: false,
                }
            }

            pub const fn provider_name(&self) -> &'static str {
                $provider_name
            }

            pub fn uri(&self) -> &str {
                &self.uri
            }

            pub const fn is_remote_connected(&self) -> bool {
                self.remote_connected
            }
        }
    };
}

define_graph_boundary!(ArangoGraph, "arangodb");
define_graph_boundary!(FalkorDBGraph, "falkordb");
define_graph_boundary!(HugeGraph, "hugegraph");
define_graph_boundary!(KuzuGraph, "kuzu");
define_graph_boundary!(MemgraphGraph, "memgraph");
define_graph_boundary!(NebulaGraph, "nebulagraph");
define_graph_boundary!(Neo4jGraph, "neo4j");
define_graph_boundary!(NeptuneGraph, "neptune");
define_graph_boundary!(NetworkxEntityGraph, "networkx");
define_graph_boundary!(RdfGraph, "rdf");
