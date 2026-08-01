# Schema versions

Four versions move independently because they answer different
questions. A change to any of them alters what stored data *means*.

| schema | version | changes when |
|---|---|---|
| Feature set | `ade558d616b58821` | a feature's formula, inputs, or units change (hashed automatically) |
| Dataset features | `4.0` | the observation/snapshot shape changes |
| Dataset records | `4.0` | the market record shape changes |
| Settlement parser | `2.0` | settlement detection or normalisation changes |

The feature set currently contains **70 features**.

## Why the feature version is a hash

It is computed from the declarations themselves — names, formulas,
inputs, units, tiers. It therefore cannot be forgotten: editing a
formula changes the version whether or not anyone remembers to bump it.
The other three are hand-maintained and checked by
`scripts/validate_schemas.py`.
