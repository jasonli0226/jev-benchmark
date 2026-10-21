# Jev recalibration (fit on val, scored on test)

Mean over test repeats. Platt = temperature scaling for multiclass.

| task | prompt | n val | method | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|---|---|
| adult | adult_raw_v1 | 300 | raw | 0.8051 | 0.0652 | 0.2755 | 0.4355 |
| adult | adult_raw_v1 | 300 | platt | 0.8076 | 0.0253 | 0.2670 | 0.4149 |
| adult | adult_raw_v1 | 300 | isotonic | 0.8003 | 0.0230 | 0.2702 | 0.4526 |
| adult | adult_natural_v1 | 300 | raw | 0.8092 | 0.0390 | 0.2703 | 0.4241 |
| adult | adult_natural_v1 | 300 | platt | 0.8071 | 0.0352 | 0.2690 | 0.4197 |
| adult | adult_natural_v1 | 300 | isotonic | 0.8033 | 0.0431 | 0.2794 | 0.6278 |
| adult | adult_contextual_v1 | 300 | raw | 0.8263 | 0.0315 | 0.2424 | 0.3834 |
| adult | adult_contextual_v1 | 300 | platt | 0.8263 | 0.0223 | 0.2414 | 0.3774 |
| adult | adult_contextual_v1 | 300 | isotonic | 0.8254 | 0.0224 | 0.2452 | 0.4251 |
| adult | adult_natural_fewshot_v1 | 300 | raw | 0.7937 | 0.0676 | 0.2713 | 0.4261 |
| adult | adult_natural_fewshot_v1 | 300 | platt | 0.8144 | 0.0165 | 0.2532 | 0.3932 |
| adult | adult_natural_fewshot_v1 | 300 | isotonic | 0.8107 | 0.0225 | 0.2555 | 0.4822 |
| adult | adult_contextual_fewshot_v1 | 300 | raw | 0.8131 | 0.0399 | 0.2575 | 0.4040 |
| adult | adult_contextual_fewshot_v1 | 300 | platt | 0.8315 | 0.0128 | 0.2314 | 0.3623 |
| adult | adult_contextual_fewshot_v1 | 300 | isotonic | 0.8334 | 0.0304 | 0.2358 | 0.4426 |
| agnews | agnews_v2 | 1000 | raw | 0.8963 | 0.0586 | 0.1651 | 0.5810 |
| agnews | agnews_v2 | 1000 | platt | 0.8963 | 0.0332 | 0.1615 | 0.3233 |
| agnews | agnews_v2 | 1000 | isotonic | 0.8958 | 0.0140 | 0.1537 | 0.3157 |
| sst2 | sst2_v1 | 300 | raw | 0.9599 | 0.0178 | 0.0639 | 0.1747 |
| sst2 | sst2_v1 | 300 | platt | 0.9633 | 0.0208 | 0.0615 | 0.1452 |
| sst2 | sst2_v1 | 300 | isotonic | 0.9618 | 0.0244 | 0.0637 | 0.1813 |
