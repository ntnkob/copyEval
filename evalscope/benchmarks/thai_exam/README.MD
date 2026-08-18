# Thai Exam QA Benchmark

Custom benchmark for Thai multiple-choice exams.

Supports:
- English choices: A/B/C/D/E
- Thai choices: ก/ข/ค/ง/จ
- Numeric choices: 1/2/3/4/5

Expected dataset format:

```json
{
  "system": "...",
  "query": "...",
  "response": "A",
  "domain": "medical"
}