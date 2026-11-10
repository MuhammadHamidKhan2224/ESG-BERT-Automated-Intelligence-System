# 🌐 Live Hugging Face ESGBERT Datasets Audit
**Evaluated on:** 3 Hugging Face Datasets (`ESGBERT/environment_data`, `ESGBERT/social_data`, `ESGBERT/governance_data`)

### 📈 Performance Metrics
- **Total Samples Evaluated:** 9
- **Model Alignment with Dataset Labels:** **44.4%**

### 📋 Detailed Sentence Breakdown
| HF Dataset | Sample Sentence | Predicted Dominant | E Score | S Score | G Score | Action Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `environment_data` | Over the course of the year, your partnership with The Wilderness Society, and your gen... | **ENVIRONMENTAL** | 99.21% | 40.71% | 0.31% | Boilerplate |
| `environment_data` | Working with community and Indigenous-led coalitions, we focused on undoing the damage ... | **ENVIRONMENTAL** | 99.72% | 21.16% | 1.41% | Boilerplate |
| `environment_data` | And together we made a more inclusive conservation movement a reality. | **SOCIAL** | 98.64% | 99.97% | 0.25% | Boilerplate |
| `social_data` | Over the course of the year, your partnership with The Wilderness Society, and your gen... | **ENVIRONMENTAL** | 99.21% | 40.71% | 0.31% | Boilerplate |
| `social_data` | Working with community and Indigenous-led coalitions, we focused on undoing the damage ... | **ENVIRONMENTAL** | 99.72% | 21.16% | 1.41% | Boilerplate |
| `social_data` | These victories demonstrate the impact of combining The Wilderness Society’s expertise ... | **SOCIAL** | 5.59% | 99.53% | 0.71% | Boilerplate |
| `governance_data` | It was an honor to stand with the Tribes and see President Biden restore the Bears Ears... | **ENVIRONMENTAL** | 51.17% | 0.02% | 0.49% | Boilerplate |
| `governance_data` | These victories demonstrate the impact of combining The Wilderness Society’s expertise ... | **SOCIAL** | 5.59% | 99.53% | 0.71% | Boilerplate |
| `governance_data` | We seek the guidance of Native American and Indigenous peoples to effectively advocate ... | **NON-ESG** | 2.01% | 25.41% | 0.49% | Boilerplate |

---
*Generated automatically by ESGBERT Automation Engine.*