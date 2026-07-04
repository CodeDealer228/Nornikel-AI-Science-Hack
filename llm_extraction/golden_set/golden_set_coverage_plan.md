# Golden set — план покрытия (64 чанка)

Составлен ДО разметки, чтобы гарантировать выполнение квот покрытия из задачи.
Источники — 18 файлов из `parsed_data/texts/` (17 статей, не пересекающихся с
`ner_re_extraction/` + документ 16, переиспользованный на НОВЫХ абзацах, + одна
презентация). Диапазоны абзацев указаны по индексам `[[N]]` внутренней разметки
(см. `source_texts/*.txt`); точные `char_start`/`char_end` вычислены скриптом при
разметке и могут отличаться от черновых диапазонов ниже на несколько абзацев в
ту или иную сторону (нарезка подгонялась под целевые ~2000–2300 символов).

Легенда категорий: **N** обычный, **E** пустой/шумный (gold = пустые
entities/relations), **C** contradicts-кандидат, **O** другой домен (не
металлургия), **P** богат числовыми Property.

| # | Источник | Абзацы (черновые) | Категория | Основной фокус |
|---|---|---|---|---|
| g01 | 01_tectonic_fidesys | 1–4 | N | Expert, Facility, affiliated_with (авторский блок) |
| g02 | 01_tectonic_fidesys | 16–19 | N,P | Process, Material — моделирование тект. нарушений |
| g03 | 01_tectonic_fidesys | 24–29 | N,P | Process, Property, Equipment — методика связей конечной жёсткости |
| g04 | 01_tectonic_fidesys | 51,55,59,63–65 | N | validated_by — верификация методики в CAE Fidesys |
| g05 | 01_tectonic_fidesys | 74–84 | **E** | список источников — gold пустой |
| g06 | 02_lukanina_basalt | 3–9 | N | Publication, Expert, Facility, described_in (EN abstract+citation) |
| g07 | 02_lukanina_basalt | 77–85 | N | Process, Equipment, Property — постановка задачи, фидер |
| g08 | 02_lukanina_basalt | 103–118 | N,P | Property (размеры фидера, табл. 1), Equipment |
| g09 | 02_lukanina_basalt | 152–168 | N,P | Property (физ. свойства среды, табл. 2, диапазоны) |
| g10 | 02_lukanina_basalt | 174–183 | N,P | Process, Property — моделирование при постоянном расходе |
| g11 | 02_lukanina_basalt | 374–388 | **E** | список литературы (EN) — gold пустой |
| g12 | 17_slag_temperature | 11–15 | N | Expert, Facility, affiliated_with (RU авторский блок) |
| g13 | 17_slag_temperature | 22–26 | N | Process, Equipment, Material — альтернативные агрегаты обеднения |
| g14 | 17_slag_temperature | 34–38 | N | Equipment, Process, Material — печь Таммана |
| g15 | 17_slag_temperature | 40–43 | N,P | Property — зависимость от температуры, табл. 1 |
| g16 | 17_slag_temperature | 60–61 | **C** | contradicts — отсутствие донной фазы при 1300/1350°C |
| g17 | 17_slag_temperature | 86–99 | **E** | список литературы — gold пустой |
| g18 | 18_rockburst_potential | 1–3 | N,**O** | Expert, Facility, affiliated_with |
| g19 | 18_rockburst_potential | 9–11 | N,**O** | Process, Property, Publication — критерий Кайзера |
| g20 | 18_rockburst_potential | 26–29 | N,**O** | Publication (ГОСТ), Experiment, Process |
| g21 | 18_rockburst_potential | 36–40 | **C**,**O** | contradicts — методика ГОСТ vs ASTM, разные % неудароопасных проб |
| g22 | 18_rockburst_potential | 58–60 | N,**O** | Process, Property — выводы |
| g23 | 19_chlorination | 1–4 | N | Expert, Facility, Material |
| g24 | 19_chlorination | 9–13 | N | Process, Material, Equipment — действующая технология |
| g25 | 19_chlorination | 15–19 | N | Equipment, Process, Experiment — методика экспериментов |
| g26 | 19_chlorination | 31–35 | N,P | Property (% извлечения), produces_output |
| g27 | 20_process_control | 1–8 | N | Expert, Facility, affiliated_with (email-блок) |
| g28 | 20_process_control | 13–18 | N | Process, Equipment — способы контроля |
| g29 | 20_process_control | 27–30 | N | Equipment, Process, Property — метод ЛОЭС |
| g30 | 20_process_control | 60–78 | **E** | список литературы — gold пустой |
| g31 | 23_rock_homogeneity | 26–33 | N,**O** | Process, Property, Material — гипотеза, условие Гриффитса |
| g32 | 23_rock_homogeneity | 36–41 | N,**O** | Process, Property — мысленный эксперимент |
| g33 | 23_rock_homogeneity | 69–72 | **C**,**O** | contradicts — аномалия мрамора Каррара |
| g34 | 23_rock_homogeneity | 82–85 | N,**O** | validated_by, Property — выводы |
| g35 | 34_yakovleva_standards | 1–2 | N | Expert, Facility |
| g36 | 34_yakovleva_standards | 7–12 | N | Material, Publication (ГОСТ), described_in |
| g37 | 34_yakovleva_standards | 55–63 | N | Material, Publication, Facility, described_in |
| g38 | 34_yakovleva_standards | 114–130 | **E** | литература — gold пустой |
| g39 | 38_affinated_metals | 1–3 | N | Expert, Facility, affiliated_with |
| g40 | 38_affinated_metals | 14–17 | N | Material, Property, Equipment — характеристика сырья |
| g41 | 38_affinated_metals | 25–30 | N,P | Process, Material, Property, produces_output |
| g42 | 40_ag_pt_analysis | 3–6 | N | Expert, Facility |
| g43 | 40_ag_pt_analysis | 24–28 | N | Process, Property, Equipment — матричные влияния |
| g44 | 40_ag_pt_analysis | 102–106 | **C** | contradicts — неудачный выбор линии Bi в ГОСТ 33730-2016 |
| g45 | 40_ag_pt_analysis | 120–130 | **E** | литература — gold пустой |
| g46 | 42_anhydrite_gypsum | 1–4 | N | Expert, Facility, affiliated_with |
| g47 | 42_anhydrite_gypsum | 14–22 | N | Material, Process, Property |
| g48 | 42_anhydrite_gypsum | 31–38 | N,P | Property (температуры ДТА), Equipment |
| g49 | 44_matte_granulation | 0–4 | N | Expert, Facility, Material, Process |
| g50 | 44_matte_granulation | 22–28 | N | Equipment, Process, Experiment — гранулятор |
| g51 | 44_matte_granulation | 44–49 | N,P | Property, uses_material — расход щёлочи |
| g52 | 45_magnetic_sep_p1 | 0–4 | N | Expert, Facility, affiliated_with |
| g53 | 45_magnetic_sep_p1 | 30–36 | N,P | Process, Property, Equipment — критерий эффективности |
| g54 | 45_magnetic_sep_p1 | 78–92 | **E** | список источников — gold пустой |
| g55 | 16_ventilation_reuse | 118–124 | N,**O** | Publication/Expert/Facility (EN abstract) — НОВЫЙ фрагмент, не пересекается с ner_re_examples.md |
| g56 | 46_magnetic_sep_p2 | 17–22 | N | Process, Property, Equipment — этапы испытаний |
| g57 | 46_magnetic_sep_p2 | 53–59 | N,P | Property (рекомендованные параметры), operates_at_condition |
| g58 | 53_catalyst_lifetime | 2–5 | N | Expert, Facility, Material, Process |
| g59 | 53_catalyst_lifetime | 31–33 | **C** | contradicts — противоположные тренды конверсии COS/H2S |
| g60 | 53_catalyst_lifetime | 89–109 | **E** | библиографический список — gold пустой |
| g61 | 54_se_te_electrowinning | 0–3 | N | Expert, Facility, Process, Material |
| g62 | 54_se_te_electrowinning | 15–20 | N,P | Property (линейная зависимость Se/Te), Process |
| g63 | 16_ventilation_reuse | 25–29 | N,**O** | Process, Equipment, Property — модель турбулентности SST — НОВЫЙ фрагмент |
| g64 | 16_ventilation_reuse | 36–40 | N,**O** | validated_by, Experiment, Property — верификация модели — НОВЫЙ фрагмент |

## Проверка квот по плану

- Сущности (мин. 8 чанков на тип): Material/Process/Equipment/Property/Experiment
  встречаются в большинстве из 58 не-пустых чанков (≫8 каждый). Publication —
  явно запланирован в g06, g19, g20, g21, g36, g37, g44, g55 (8 чанков uже по
  плану; дополнительные вхождения ожидаются по факту разметки — методические
  разделы часто ссылаются на ГОСТ/патенты в самом тексте, не только в списке
  литературы). Expert/Facility — во всех авторских блоках (g01,g12,g18,g23,
  g27,g35,g39,g42,g46,g49,g52,g55,g58,g61 = 14 чанков).
- Отношения (мин. 4 чанка на тип): `uses_material`, `operates_at_condition`,
  `produces_output`, `described_in` — многократно в N-чанках. `validated_by`:
  g04, g20, g34, g64 (+ ожидаются доп. по факту). `contradicts`: g16, g21, g33,
  g44, g59 (5, с запасом). `affiliated_with`: все авторские блоки (14 чанков).
  `authored_by`: те же авторские блоки, где есть явное название публикации
  (g06, g55 точно; остальные — Expert аффилирован с Facility, но не всегда
  явно "автор статьи X" — доп. authored_by добавляются по факту разметки).
- Property с числовыми диапазонами/операторами (мин. 6): g08, g09, g15, g26,
  g41, g48, g51, g57, g62 (9, с запасом).
- Contradicts (мин. 4): g16, g21, g33, g44, g59 (5, с запасом одного на случай,
  если один из кандидатов при ближайшем чтении не подтвердится как настоящее
  противоречие).
- Пустые/шумные (мин. 6, ровно): g05, g11, g17, g30, g38, g45, g54, g60 — это
  **8** запланировано (небольшой запас, часть может быть переиспользована как
  "не хватило другого материала" по ходу разметки, но не менее 6 останется).
- Другой домен, документ 16 (мин. 3): g55, g63, g64 — ровно 3 из документа 16
  (плюс g18, g19, g20, g21, g22 из горной геомеханики/удароопасности и g31–g34
  из другой статьи о горных ударах, и g01–g04 из тектонического моделирования
  — все тоже вне металлургии, для дополнительной уверенности сверх минимума).

Фактическое покрытие после разметки — в `golden_set_validation_report.md`,
посчитано программно по итоговому `golden_set.jsonl`, а не по этому плану.
