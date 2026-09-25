# Coursework provenance

This public repository was reconstructed from the submitted project report and the available S2/S3/S4 scripts.

## Important provenance note

The file named `S1.py` provided during repository cleanup was unrelated to the image-processing assignment (it was a PDF-to-PowerPoint utility). Therefore, the public Gaussian-smoothing chapter was reconstructed from the report rather than copied from that file.

The reconstructed chapter preserves the report's stated design:
- manual RGB-to-gray conversion,
- 2D Gaussian kernel construction,
- kernel size based on approximately 3 sigma,
- kernel normalization,
- convolution without cv2.filter2D/scipy.signal.convolve2d,
- sigma values 0.5, 1.0, 2.0, and 4.0,
- separable X-then-Y Gaussian filtering,
- direct comparison between 2D and separable outputs.

The report documented maximum absolute differences on the order of 1e-16 between the two Gaussian implementations.

## Public-repo correction to the original comparison stage

The original S4 script compared saved PNG renderings from earlier stages. The public version compares the raw numerical arrays directly, so plot titles, margins, rendering scale, and figure backgrounds cannot contaminate the numerical difference images.
