# Single-Sequence Examples

The HyFI repository includes example datasets and configuration files to help you get started. The datasets below are examples of analysis of individual earthquake sequences (single-sequence workflow).

---

## St. Léonard (Truttmann et al., 2023)

Configuration file: `./config/config_example_StLeonard.json`

### Running the example

In the terminal, navigate to the HyFI directory (needs to be adjusted to the path where your HyFI directory sits locally, e.g. `cd HyFI`)

Execute HyFI with the corresponding configuration file
```bash
hyfi run -c config/config_example_StLeonard.json 
```

In the HyFI directory, you should now see a folder `./output_StLeonard` where all the results are stored.

---

## References

-  Truttmann, S., Diehl, T., & Herwegh, M. (2023). Hypocenter-based 3D imaging of active faults: Method and applications in the Southwestern Swiss Alps. Journal of Geophysical Research: Solid Earth, 128, e2023JB026352. https://doi.org/10.1029/2023JB026352 

---

Happy fault imaging! 🎉