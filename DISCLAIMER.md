# DISCLAIMER

## Important Notice

**This software is provided for research and educational purposes only.**

## Limitations and Risks

### Uncertainty Estimates
- Uncertainty estimates provided by this software may be **unstable**, **misleading**, or **incorrect**
- These estimates are **not guaranteed** to accurately reflect true model uncertainty
- Results may vary significantly across different runs, datasets, and configurations

### Model Predictions
- Model predictions should **not be used** for making regulated decisions without human review
- The software is **not intended** for use in safety-critical applications
- Users must **validate all results** before applying to real-world problems

### Calibration and Reliability
- Calibration metrics may not reflect true model reliability
- Uncertainty calibration can be **poor** even when metrics appear good
- Model confidence does **not guarantee** prediction accuracy

## Appropriate Use Cases

### Research Applications
- Academic research in uncertainty quantification
- Method development and comparison
- Educational demonstrations and tutorials

### Educational Use
- Learning about Bayesian neural networks
- Understanding uncertainty estimation methods
- Exploring calibration and reliability concepts

## Inappropriate Use Cases

### Production Systems
- **Do not use** in production environments without extensive validation
- **Do not use** for automated decision-making systems
- **Do not use** in safety-critical applications

### Regulated Domains
- **Do not use** for medical diagnosis without human oversight
- **Do not use** for financial trading decisions
- **Do not use** for autonomous vehicle control
- **Do not use** for any regulated decision-making process

## User Responsibilities

### Validation Required
- Users must **validate all results** independently
- Users must **consult domain experts** before applying to real problems
- Users must **understand the limitations** of uncertainty estimation methods

### Risk Assessment
- Users must **assess risks** associated with their specific use case
- Users must **implement appropriate safeguards** and human oversight
- Users must **not rely solely** on automated uncertainty estimates

### Compliance
- Users must **ensure compliance** with applicable regulations
- Users must **obtain necessary approvals** for regulated applications
- Users must **follow best practices** for responsible AI deployment

## Technical Limitations

### Method Limitations
- Monte Carlo Dropout may underestimate uncertainty
- Variational methods may be poorly calibrated
- Deep ensembles are computationally expensive
- All methods may fail on out-of-distribution data

### Implementation Limitations
- Code may contain bugs or implementation errors
- Hyperparameters may not be optimal for all use cases
- Performance may vary across different hardware and software environments
- Results may not be reproducible due to random variations

## No Warranty

**This software is provided "as is" without warranty of any kind.**

- No warranty of accuracy, reliability, or fitness for any purpose
- No warranty that the software will meet your requirements
- No warranty that the software will be error-free or uninterrupted
- No warranty regarding the results obtained from using the software

## Limitation of Liability

**The authors and contributors shall not be liable for any damages arising from the use of this software.**

This includes but is not limited to:
- Direct, indirect, incidental, or consequential damages
- Loss of profits, data, or business opportunities
- Damages resulting from incorrect predictions or uncertainty estimates
- Damages resulting from inappropriate use of the software

## Recommendations

### For Researchers
- Always validate results on multiple datasets
- Compare with established baselines
- Report limitations and failure cases
- Follow best practices for reproducible research

### For Educators
- Emphasize limitations and appropriate use cases
- Teach critical evaluation of uncertainty estimates
- Include discussions of responsible AI practices
- Provide context for real-world applications

### For Practitioners
- Never use without human oversight
- Always validate on domain-specific data
- Implement appropriate safety measures
- Consult with domain experts
- Follow regulatory requirements

## Contact

If you have questions about appropriate use cases or need clarification on limitations, please:
- Open an issue on the project repository
- Contact the development team
- Consult with domain experts in your field

## Version

This disclaimer applies to all versions of the software. Users are responsible for reviewing the disclaimer for any updates.

---

**By using this software, you acknowledge that you have read, understood, and agree to the terms of this disclaimer.**
