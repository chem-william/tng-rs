#[derive(Debug)]
pub struct Bond {
    /// One of the atoms of the bond
    from_atom_id: i64,
    /// The other atom of the bond
    to_atom_id: i64,
}
