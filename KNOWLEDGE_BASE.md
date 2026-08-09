# Fishometry Pipeline Knowledge Base

## 1. Purpose

Fishometry studies whether the physical length of a fish can be estimated from a single photograph without placing a ruler, calibration board, or other known-size reference beside the fish.

This is difficult because an image records appearance in pixels rather than physical units. The same fish can occupy very different numbers of pixels depending on camera distance, focal length, crop, orientation, and perspective. Conversely, two fish of different physical lengths can occupy similar areas of an image.

The pipeline addresses this ambiguity by combining several kinds of evidence:

- The visible dimensions and proportions of the fish.
- The positions and sizes of the head, tail, body, and sometimes the eye.
- Relative depth patterns within the image.
- The outline and shape of the segmented fish.
- Visual appearance from an isolated fish image.
- The scene in which the photograph was taken.
- Fish type when that information is available.

The project is experimental rather than a single final predictor. It creates many controlled combinations of datasets, feature families, depth information, and prediction methods so their effects can be compared consistently.

## 2. Core Research Questions

The work is organized around these questions:

1. How accurately can fish length be estimated from controlled photographs?
2. How much does performance change in uncontrolled outdoor scenes?
3. Do models benefit from relative measurements instead of raw pixel sizes?
4. Does relative depth improve length estimation?
5. Do segmentation-based shape measurements improve predictions?
6. Does visual scene context help explain outdoor variation?
7. Does fish type improve global models, and are separate type-specific models better?
8. Do image-based models add information beyond engineered measurements?
9. How sensitive are results to apparent zoom and image scale?

Every major pipeline stage supports one or more of these questions.

## 3. Dataset Families

### Controlled indoor dataset

The controlled dataset contains laboratory photographs with comparatively consistent capture conditions. Each record connects one image to a measured fish length.

This dataset provides the clearest setting for studying the relationship between image geometry and physical length. Because the scene varies less, changes in visible fish size and shape are less confounded by background, pose, and capture conditions.

The controlled measurements also include the eye, enabling dedicated experiments with it as a possible scale cue.

### Outdoor multi-type dataset

The outdoor dataset contains photographs captured under heterogeneous real-world conditions. Images can differ in:

- Camera distance and viewpoint.
- Fish orientation.
- Lighting and weather.
- Background surface and scene depth.
- Presence of people, fishing equipment, nets, or other objects.
- Fish type and natural body shape.
- Image resolution, crop, and compression.

Each record connects an image to a measured length and a fish type. The type label supports type-aware splitting, type-aware features, type-level analysis, and specialized predictors.

This dataset tests whether the approach generalizes when image geometry alone is not enough to explain physical scale.

### Zoom-derived controlled dataset

The zoom-derived dataset is created from the controlled dataset after its train, validation, and test assignments have been fixed.

Each usable source image produces three related observations:

1. A re-encoded version of the original image.
2. A zoom-in version created by taking a centered crop and resizing it back to the original dimensions.
3. A zoom-out version created by reducing the image dimensions.

The physical length label and split assignment are copied from the source image to all three observations.

This inheritance is essential. Related versions of one fish must never be placed in different partitions. If one version were used for training and another for testing, the test result would be contaminated by near-duplicate information.

The derived dataset is three times the source size when every source image is readable. It does not introduce new fish or new physical measurements. It introduces controlled variation in apparent scale.

## 4. Pipeline Overview

The complete information flow is:

1. Assemble images and ground-truth metadata.
2. Assign source observations to training, validation, and test partitions.
3. Optionally derive zoom variants while preserving source assignments.
4. Locate fish body parts and the full fish in each image.
5. Normalize orientation when alignment is enabled.
6. Measure the aligned image again.
7. Estimate relative depth.
8. Separate the fish from its background.
9. Create a standardized isolated-fish image.
10. Attach original-scene context when available.
11. Engineer numerical and categorical features.
12. Fit a matrix of baseline, tabular, neural, and image-based predictors.
13. Generate comparable predictions for every surviving observation.
14. Evaluate errors by partition, model, fish type, length range, and individual image.

The stages are sequential. A later stage assumes that the required output of each earlier stage exists and remains consistent with the same image and split assignment.

## 5. Source Data Contract

Every source observation requires:

- A unique image identity.
- A measured fish length.
- A readable image corresponding to the identity.
- A fish type when type-aware behavior is enabled.

The length unit must be consistent within a dataset. The pipeline learns a mapping to whatever unit the labels use; it does not infer or convert units.

Records with missing required metadata are excluded before splitting. Unique image identity is important because it connects metadata, intermediate artifacts, predictions, and visual inspection throughout the workflow.

### Data provenance

Both dataset families were supplied by the thesis instructor, who works directly with the people who catch and photograph the fish. The controlled indoor collection was led by the instructor, which is why its capture conditions are comparatively uniform and why it carries an eye annotation at all.

The outdoor photographs come from that same network of contributors rather than from a single controlled session. Their heterogeneity is therefore a property of how the images were really obtained, not an artificial perturbation added for the study.

### Length definition

The target is a straight-line total length: from the frontmost point of the head to the rearmost point of the tail. It is not a curved measurement along the body, and it is not a fork or standard length.

Each length was taken with a measuring tape by the person who photographed the fish. Two consequences follow:

- The measurement procedure was not centrally supervised, so tape placement, tail compression, and reading precision vary between contributors.
- Label error of this kind is irreducible from the images alone and places a ceiling on any achievable accuracy, discussed further under measurement quality.

Because the label is a straight-line head-to-tail distance, the geometric features that describe the same straight-line extent in pixels are the ones expected to carry most of the signal.

## 6. Train, Validation, and Test Design

### Training partition

The training partition is used to estimate model parameters. Baseline averages, linear relationships, tree structures, neural weights, image representations, and specialized type-level relationships are learned from these observations.

### Validation partition

The validation partition supports model selection for methods that monitor performance during fitting. It can determine when a boosted or neural model should stop improving and which saved neural state should be retained.

It is not treated as final evidence of generalization because it can indirectly influence model choice.

### Test partition

The test partition is held out from parameter fitting and validation-based selection. It provides the most appropriate estimate of performance on unseen observations from the same data-generating setting.

### Global and type-aware splitting

Controlled data is divided as one population.

Outdoor data is divided separately within every fish type. The type-level partitions are then combined. This preserves representation of each type across training, validation, and test data as far as the available sample size permits.

Very small type groups can still produce empty or unstable partitions because whole observations cannot be divided fractionally.

### Split stability

Once downstream processing or experiments begin, the split manifest becomes part of the experiment definition. Reassigning observations changes the training evidence and the evaluation population, which makes old and new metrics directly incomparable.

Derived observations always inherit the source split. This protects evaluation from duplicate-family leakage.

## 7. Detection and Landmark Measurement

The first image-analysis task is to locate biologically meaningful regions:

- The complete fish.
- The head.
- The tail.
- The eye when supported by the controlled dataset.

For each detected region, the pipeline records the rectangular boundaries, width, and height. It also records image dimensions.

These measurements serve several purposes:

- Head and tail centers define the fish's dominant direction.
- The full-fish box gives visible size and aspect information.
- The eye supplies an additional possible scale cue in controlled images.
- Region centers identify where relative depth should be sampled.
- Head and tail points guide segmentation.

An image must contain a usable and unambiguous set of required detections to continue. Rejecting uncertain observations reduces sample count but prevents downstream measurements from being built on missing or contradictory landmarks.

### Trained detectors

The landmark detectors are not off-the-shelf models. Two object detectors were trained for this work, one per dataset family, on the same photographs the study analyses:

- The controlled detector locates the head, tail, eye, and whole fish.
- The heterogeneous outdoor detector locates the head, tail, and whole fish.

Training two detectors rather than one reflects the two capture settings. It is also the reason the eye is available only for controlled data: the outdoor detector has no eye class, so no outdoor eye feature can exist regardless of what an individual photograph shows.

Both were trained for 100 epochs at 640-pixel input with a fixed seed of 0.

| Detector | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Epochs | Batch size | Input size | Seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled | 99.632 | 100.0 | 99.5 | 91.666 | 100 | 16 | 640 | 0 |
| Heterogeneous outdoor | 85.355 | 85.867 | 88.34 | 80.585 | 100 | 27 | 640 | 0 |

The gap between the two is a substantive result in itself. On controlled images the landmarks are located almost perfectly, so the geometric features downstream can be treated as a faithful description of the photograph, and length errors can be attributed to the length-estimation step rather than to detection.

Outdoor detection is markedly weaker. Roughly one in seven required landmarks is missed or misplaced, and localization quality at stricter overlap thresholds falls further. This has two effects on every outdoor result:

- Attrition. Images whose required landmarks are not detected confidently never reach the processed population, so outdoor sample loss is partly a detector property.
- Feature noise. Among surviving images, box edges are less precise, which adds noise to relative width, height, area, and aspect, and shifts the points at which depth is sampled.

Outdoor length errors therefore combine detector error with estimator error, and controlled-versus-outdoor comparisons should not be read as a pure statement about scene difficulty.

## 8. Orientation Normalization

Fish may appear left-to-right, right-to-left, diagonal, or nearly vertical. Raw width and height are difficult to compare when orientation changes.

Alignment uses the vector from the tail center to the head center. The image is rotated until that direction is horizontal. The rotation canvas is expanded to avoid clipping, orientation is corrected when necessary, and border regions introduced by rotation are cropped away.

The original detection coordinates are not reused after this transformation. Body parts are detected again in the aligned image so all later measurements share the new coordinate system.

Alignment aims to reduce pose variation, not to change the scene's meaning. Scene context continues to describe the original photograph.

## 9. Final Detection

After alignment, a second measurement pass locates the fish, head, tail, and optional eye in the image used by downstream geometry.

This pass produces the definitive pixel measurements for feature creation. Re-measurement is preferable to transforming old boxes because rotation, expansion, flipping, and cropping can compound coordinate errors.

When alignment is disabled, the same raw-image detections are used as the final measurements.

## 10. Relative Depth Estimation

A monocular depth estimator produces a dense relative-depth value for every pixel. These values describe which parts of one image appear nearer or farther according to learned visual cues.

They are not measurements in centimeters and cannot be treated as absolute camera distance without calibration.

The pipeline samples median values around three locations:

- Head center.
- Fish-body center.
- Tail center.

Using a small neighborhood median is more robust than using a single pixel, which may lie on an edge or contain local noise.

Depth experiments test whether relative three-dimensional cues explain length variation that two-dimensional box geometry cannot.

The complete depth field is also retained as an intermediate visual artifact so the spatial estimate can be inspected.

## 11. Fish Segmentation

Detection supplies boxes, but a box includes background and does not describe the true outline of the fish. Segmentation uses head and tail locations as positive prompts to produce a binary fish mask.

The largest external contour is treated as the primary fish region. From it, the pipeline derives:

- Mask area.
- Perimeter.
- Major and minor axes of an approximate ellipse.
- Solidity, which compares the fish contour with its convex hull.

These features capture body form more directly than a rectangular box. They can help distinguish long narrow fish from shorter deep-bodied fish that occupy similar box areas.

The binary mask is retained as an intermediate artifact for reuse and inspection.

## 12. Isolated Fish Images

The segmentation mask is applied to the image so all background pixels become black. The visible fish is tightly cropped, resized while preserving aspect ratio, and centered on a fixed square canvas.

This standardized image serves the image-based predictor. It reduces the opportunity for that model to learn accidental background correlations and gives every observation a common input size.

The isolated image retains color, texture, and fine shape information that engineered numerical features may not express.

## 13. Original-Scene Context

Outdoor photographs contain scale-related and domain-related cues outside the fish itself. These cues are not measured geometrically. They are read out of the original photograph by a vision-language model, which is asked to answer a fixed set of questions with a fixed set of permitted answers. The pipeline can attach structured context describing:

- Whether the background extends into distant scenery or is a nearby surface.
- Whether other objects are visible.
- Whether the fish is in or on a net.
- The surface or situation in which the fish is positioned.
- The fish's original direction within the frame.
- The lighting condition.

These values are extracted from the original photograph because rotation is an analytical transformation, not the scene in which the photograph was captured.

Previously obtained context can therefore be reused alongside newly measured aligned geometry. This preserves the intended distinction between scene information and fish-shape information.

Context categories are retained as structured attributes. Numerical context indicators are used when they are available in the processed dataset.

Constraining the model to a closed set of categories keeps the resulting columns comparable across images and directly usable as features. It does not make them observations. Each context value is a judgement about the photograph, produced with no reference to the ground-truth length, and it carries its own error rate. A scene can be labelled with the wrong surface, the wrong lighting, or the wrong background depth, and nothing downstream detects that. Context features should be interpreted as automatic annotation of the scene rather than as measurement of it.

## 14. Engineered Feature Families

The prediction experiments use named feature bundles. A bundle determines which measurements are supplied together, but its availability depends on the dataset family.

Three distinctions are important:

1. The eye bundle exists only for the controlled and zoom-derived controlled datasets. It does not exist for outdoor data.
2. The rich shape-and-context bundle exists only for outdoor experiments. It is not part of the controlled or zoom-derived experiment matrix.
3. For an image-based method, the bundle name describes the auxiliary measurements added to the image. It does not mean that the image has been removed from the method.

### Relative coordinate features

Relative geometry compares the fish box with the containing image:

- Fish width as a share of image width.
- Fish height as a share of image height.
- Fish box area as a share of image area.
- Fish width-to-height aspect ratio.
- A square-root area measure that expresses box scale in a length-like form.

Relative features reduce direct dependence on image resolution, although they cannot fully resolve camera distance or perspective.

This bundle uses the full-fish box and image dimensions. It does not include eye size, segmentation shape, scene context, or sampled depth unless depth is explicitly added as a separate experiment condition.

The coordinate bundle is available for all three dataset families.

### Eye features

The eye bundle contains four direct pixel measurements:

- Eye width.
- Eye height.
- Full-fish width.
- Full-fish height.

The eye is investigated as a possible biological scale cue. These values are not the same as the relative coordinate bundle: they are direct detected dimensions rather than fish-to-image ratios.

Eye measurements are available only for the controlled dataset and its zoom-derived counterpart. The outdoor detector does not provide an eye measurement, so there are no outdoor eye-feature prediction results.

The eye bundle does not include segmentation shape or scene context. Sampled depth is appended only in the eye-plus-depth experiments.

### Shape and context features

The rich outdoor bundle combines:

- Relative coordinate features.
- Segmentation area, perimeter, axes, and solidity.
- Background-depth category.
- Indicators for other objects and fishnet presence.
- Encoded placement, orientation, and available scene categories.

This bundle does not include eye measurements. It is available only for outdoor data because that experiment family is designed to study uncontrolled scenes and fish-type variation.

The rich bundle is deliberately compared with the smaller coordinate bundle. This reveals whether mask shape and scene context add predictive information beyond apparent fish size and aspect alone.

In prediction terminology, "features" refers specifically to this rich outdoor bundle. It is not a generic label meaning every possible input.

### Fish-type features

When fish type is known, it is represented through separate binary indicators. This allows a global model to learn different offsets or relationships for different body forms while still using the complete population.

Fish-type indicators are included automatically in every outdoor tabular bundle, including the outdoor coordinate bundle. Therefore an outdoor coordinate result uses relative geometry plus fish-type identity, whereas a controlled coordinate result uses relative geometry without fish type.

In a separately fitted fish-type model, the data has already been restricted to one type. The type indicator is then constant and specialization comes from fitting a separate relationship for that type.

### Depth additions

Depth-enabled experiments append the available sampled and derived depth values to the selected base family.

The appended depth group represents:

- The sampled head region.
- The sampled body region.
- The sampled tail region.
- A derived signed depth field.
- A derived absolute depth field.

Depth is not a standalone base bundle. It is a second experimental switch applied to eye, coordinate, or rich inputs. This creates pairs such as coordinate versus coordinate-plus-depth and eye versus eye-plus-depth.

Running every base family both with and without depth creates a direct ablation: any consistent performance difference can be attributed to the inclusion of depth information, subject to ordinary experimental uncertainty.

### Dataset feature availability

| Input or bundle | Controlled | Zoom-derived controlled | Outdoor |
| --- | --- | --- | --- |
| Relative coordinate bundle | Yes | Yes | Yes |
| Eye bundle | Yes | Yes | No |
| Rich shape-and-context bundle | No | No | Yes |
| Fish-type indicators | No | No | Yes |
| Optional depth addition | Yes | Yes | Yes |
| Standardized isolated-fish image | Used by the image-plus-measurement method | Used by the image-plus-measurement method | Used by the image-plus-measurement method |
| Aligned full-color image representation | No | No | Used by the outdoor embedding method |

This availability table defines which prediction combinations can exist. A missing combination is intentional when the required measurement is not available or is not part of that dataset's experiment design.

## 15. Complete-Case Handoff

Not every source observation reaches training. An observation can be excluded because of missing metadata, missing images, incomplete detections, failed alignment, unavailable depth, unusable segmentation, or incomplete required context.

This means the processed population is a selected subset of the original split population.

That selection has two consequences:

1. All compared models operate on records with complete required inputs, which supports fair within-table comparison.
2. Reported performance describes the observations that the preprocessing system could successfully process, not necessarily every photograph encountered in practice.

Attrition should be measured and reported by dataset and split. A highly accurate model is not a complete system if preprocessing rejects a large or systematically biased portion of the data.

## 16. Experiment Matrix

Training is designed as a factorial comparison rather than a single run.

The main axes are:

- Dataset family.
- Feature family.
- Depth included or excluded.
- Prediction method.
- Global or fish-type-specific fitting when types are available.

Every completed experiment produces one predicted length for each surviving observation. Keeping these predictions side by side makes comparisons use the same target and split membership.

### How to read one experiment

Every result can be understood through four questions:

1. Which prediction method produced it?
2. Which base feature bundle was supplied?
3. Was the depth addition included?
4. Was one global relationship fitted, or were separate fish-type relationships fitted?

The dataset family supplies a fifth piece of context because it determines which feature bundles and specialization choices are possible.

### Controlled experiment inventory

The controlled dataset uses two base bundles:

- Eye measurements.
- Relative coordinates.

Each bundle is run without depth and with depth. This gives four input conditions for each core prediction method:

1. Eye.
2. Eye plus depth.
3. Coordinates.
4. Coordinates plus depth.

The four core methods are linear regression, gradient-boosted trees, a tabular neural network, and the image-plus-measurement model.

| Controlled result group | Calculation | Prediction results |
| --- | ---: | ---: |
| Mean baseline | One training mean | 1 |
| Linear regression | Two bundles times two depth conditions | 4 |
| Gradient-boosted trees | Two bundles times two depth conditions | 4 |
| Tabular neural network | Two bundles times two depth conditions | 4 |
| Image-plus-measurement model | Two auxiliary bundles times two depth conditions | 4 |
| Total | Baseline plus sixteen fitted combinations | 17 |

The zoom-derived controlled dataset uses exactly the same experiment inventory, so it also produces 17 prediction results. The difference is the training and evaluation population, not the feature or method matrix.

### Outdoor experiment inventory

The outdoor dataset uses two different base bundles:

- Relative coordinates, with fish-type indicators.
- Rich shape-and-context features, also with fish-type indicators.

Each bundle is again run without and with depth. The same four core methods first fit global models, creating sixteen global prediction results.

Those sixteen combinations are then repeated with separate fitting for each fish type. The final output still contains sixteen fish-type-specific result columns, not one result column per method per type. For any one of these columns, each row receives the prediction from the model fitted for that row's fish type.

If there are T retained fish types, these sixteen result columns represent sixteen times T underlying fitted models.

The two outdoor image-embedding families each add four results: coordinates, coordinates plus depth, rich features, and rich features plus depth. Their type specialization is handled internally and does not add another visible set of result columns.

| Outdoor result group | Calculation | Prediction results |
| --- | ---: | ---: |
| Fish-type mean baseline | One type-aware baseline result | 1 |
| Four global core methods | Four methods times two bundles times two depth conditions | 16 |
| Four separately fitted fish-type methods | Four methods times two bundles times two depth conditions | 16 |
| EfficientNet and DINOv2 embedding regression | Two methods times two bundles times two depth conditions | 8 |
| Total | Baseline plus forty non-baseline result columns | 41 |

This is why the outdoor prediction table is substantially wider than the controlled tables. It compares feature richness, depth, method family, and global versus specialized fitting in the same result set.

### Main comparisons enabled by the matrix

- Eye versus relative coordinates on controlled data.
- The same controlled matrix before and after zoom-derived scale variation.
- Coordinates versus rich shape-and-context features on outdoor data.
- Every base bundle without depth versus the same bundle with depth.
- Linear, tree, tabular-neural, and image-plus-measurement methods using comparable auxiliary inputs.
- Global outdoor fitting versus separate fish-type fitting.
- End-to-end isolated-image learning versus fixed full-image representations with regularized regression.

## 17. Prediction Methods

The feature bundle and the prediction method are separate choices. The table below shows what each method consumes before the individual methods are described.

| Method | Image input | Measurement input | Dataset scope |
| --- | --- | --- | --- |
| Mean baseline | None | No engineered features; training length only | All datasets |
| Linear regression | None | Selected eye, coordinate, or rich bundle; optional depth | All configured bundles |
| Gradient-boosted trees | None | Selected eye, coordinate, or rich bundle; optional depth | All configured bundles |
| Tabular neural network | None | Selected eye, coordinate, or rich bundle; optional depth | All configured bundles |
| Image-plus-measurement model | Standardized isolated-fish image | Selected auxiliary bundle; optional depth | All configured bundles |
| Image-embedding regression | Aligned full-color image representation | Outdoor coordinate or rich bundle; optional depth and fish type | Outdoor only |

### Mean baseline

The baseline predicts the average training length.

For multi-type data, it predicts the average training length within each fish type. This is an important reference because a complex model should outperform a predictor that knows only the typical size of each type.

The baseline does not use coordinates, eye size, depth, segmentation, context, or image appearance. It produces one result per dataset rather than one result per feature condition.

Its purpose is to reveal how much performance comes merely from knowing the typical size of the training population or, outdoors, the typical size of each fish type.

### Linear regression

Linear regression estimates one additive relationship between selected features and length. It provides a transparent reference for whether the engineered measurements have a simple approximately linear association with physical size.

For controlled data it is fitted separately with eye, eye-plus-depth, coordinates, and coordinates-plus-depth. For outdoor data it is fitted with coordinates, coordinates-plus-depth, rich features, and rich-features-plus-depth. Outdoor versions also receive fish-type identity.

Linear regression uses no image pixels. A linear coordinate result is therefore genuinely a measurement-only result.

Its main strength is interpretability and its main limitation is that one additive relationship may not represent complex interactions among apparent scale, fish shape, depth, and type.

### Gradient-boosted trees

The boosted-tree predictor can represent nonlinear thresholds and feature interactions. It is useful when the effect of apparent size, type, shape, or depth changes across the feature range.

Validation observations guide when boosting should stop.

It runs over the same dataset-specific feature combinations as linear regression and also uses no image pixels. A coordinate tree and a rich-feature tree differ only in the measurement bundle supplied to the tree method.

Trees can capture relationships such as one apparent-size effect for a narrow fish and another for a deep-bodied fish. Their greater flexibility can improve performance, but it can also fit dataset-specific patterns that do not generalize.

### Tabular neural network

The tabular neural predictor transforms the selected numerical features through multiple nonlinear layers. It is trained on the training partition and retains the state with the best validation loss.

It receives the same four input conditions available to the linear and tree methods for the selected dataset. It does not inspect image pixels.

The method tests whether a learned nonlinear combination of geometry, depth, shape, context, and fish type is more useful than linear addition or tree partitions. Its flexibility also makes it more sensitive to initialization, sample size, and validation stability.

### Image-plus-measurement model

The image model learns appearance features from isolated fish images. Those visual features are combined with the same selected auxiliary measurements used by the tabular experiments, then mapped to length.

This tests whether color, texture, fine contour, and unmodeled visual structure add useful information beyond engineered features.

The named feature bundle describes only its auxiliary branch:

- An image-plus-eye result uses the isolated fish image together with eye and fish dimensions.
- An image-plus-coordinates result uses the isolated fish image together with relative geometry.
- An image-plus-rich-features result uses the isolated fish image together with outdoor geometry, segmentation shape, and context.
- A depth-enabled version adds depth values to the corresponding auxiliary branch.

Therefore an image-plus-coordinates result is not a coordinates-only prediction. It always includes learned visual information from the isolated fish image.

The isolated image removes most of the scene background, so this method focuses on fish appearance. Scene information can still enter through the rich auxiliary bundle in outdoor experiments.

### Image-embedding regression

The outdoor embedding experiments extract fixed high-dimensional representations from aligned and isolated color images, combine them with engineered measurements and fish type, then fit regularized regression. One family uses ImageNet-pretrained EfficientNet-B3; the stronger family uses multiple pinned DINOv2 backbones, resolutions, framings, and image sources.

Both families include a global relationship and specialized type-level relationships when enough training observations are available for a type. Types with insufficient data fall back to the global relationship.

Each family creates four outdoor results by combining its image representation with coordinates, coordinates-plus-depth, rich features, and rich-features-plus-depth. Derived geometry and depth values are included only when their experiment label permits them.

Like the image-plus-measurement method, each bundle label names the auxiliary tabular measurements rather than the full input. Every embedding result still contains image information. Aligned-image views can retain appearance and residual scene context, while isolated-image views emphasize the fish.

### Separate fish-type models

For multi-type data, the main tabular and image-plus-measurement experiments are also repeated independently for each type.

These runs answer a different question from type indicators in a global model:

- A global model with type indicators shares most relationships across types.
- A separate type model learns all relationships only from that type.

Specialization may capture distinct body forms, but it also reduces the number of training and validation observations available to each model.

Separate fitting is applied to linear regression, boosted trees, the tabular neural network, and the image-plus-measurement method. It is not applied to controlled data because controlled records do not carry fish-type labels.

The outdoor embedding method handles specialization differently: it keeps a global fallback and uses a specialized relationship only when a fish type has enough training observations.

## 18. Split Use During Training

All methods use training observations for parameter fitting.

Validation use differs by method:

- The mean and linear baselines do not need validation-based selection.
- Boosted trees use validation to control boosting.
- Neural predictors use validation loss to retain their best state and, where applicable, stop training.
- The embedding regression selects regularization within the training process and does not use the external validation partition.

Test observations are not used to fit or select these models.

After fitting, predictions are generated for training, validation, and test observations. The split membership travels with each prediction so analysis can select the correct evaluation population.

## 19. Prediction Output

The final prediction table contains one observation per processed image and preserves:

- Image identity.
- Ground-truth length.
- Split membership.
- Fish type when available.

Each experiment adds one predicted-length column. The experiment identity communicates the prediction family, feature family, depth inclusion, and whether fitting was type-specific.

A result identity can therefore be read as:

**prediction method + feature bundle + optional depth + optional fish-type scope**

Examples in plain language are:

- Linear regression using controlled eye measurements without depth.
- A boosted-tree model using outdoor rich features plus depth.
- An image-plus-measurement model using an isolated fish image, relative coordinates, and depth.
- A separately fitted fish-type neural model using outdoor coordinates.
- Outdoor image-embedding regression using an aligned image and rich auxiliary features.

The phrase "using coordinates" must be interpreted with the method:

- For linear regression, boosted trees, and the tabular neural network, it means the coordinate measurement bundle, plus fish type outdoors.
- For the image-plus-measurement method, it means an isolated fish image plus coordinate measurements.
- For outdoor image-embedding regression, it means an aligned full-color image representation plus coordinate measurements and fish-type routing.

Likewise, eye results exist only for controlled and zoom-derived controlled data, while rich-feature results exist only for outdoor data.

The table is intentionally wide because it supports direct row-by-row comparison of all experiments without joining separate result sets during analysis.

The prediction table is the primary experiment result. Saved model states support later inspection or reuse, but the common table is what allows complete comparisons across methods and splits.

The controlled and zoom-derived controlled tables each contain 17 prediction results. The outdoor table contains 41. These counts refer to result columns; the number of underlying fitted models is larger for outdoor fish-type-specific experiments.

## 20. Evaluation Metrics

### Mean absolute error

Mean absolute error is the average absolute difference between predicted and measured length.

It uses the same unit as fish length and answers: on average, how far is a prediction from the measurement?

Lower is better.

### Mean absolute percentage error

Mean absolute percentage error divides each absolute error by the measured length and expresses the average as a percentage.

It supports comparison across fish of different sizes, but it gives greater influence to errors on small measured values and is undefined when a measured value is zero.

Lower is better.

### Coefficient of determination

R2 compares squared prediction error with the variation obtained by always predicting the mean.

- A value near 1 indicates that predictions explain most observed variation.
- A value near 0 indicates performance similar to predicting the mean.
- A negative value indicates performance worse than that reference on the selected observations.

Higher is better.

### Residual and absolute error

The residual is predicted length minus measured length. Its sign shows whether the model overestimates or underestimates.

Absolute error removes direction and shows error magnitude for an individual observation.

## 21. Evaluation Discipline

Metrics over all partitions combine observations seen during fitting with held-out observations. They are useful for diagnosing fit but must not be reported as final generalization performance.

Final model comparison should emphasize the test partition and should state:

- Dataset family.
- Selected feature family.
- Whether depth was included.
- Whether the model was global or type-specific.
- Number of evaluated observations.
- Attrition between the source split and processed population.
- Fish-type composition when applicable.

Validation metrics can support model choice. Training metrics can reveal underfitting or overfitting. Neither replaces held-out test metrics.

## 22. Visualization and Analysis Workflows

### Data explorer

The explorer connects one observation's metadata with its original image, aligned image, annotated detections, depth visualization, isolated-fish image, and available predictions.

It is used to verify that numerical results correspond to sensible intermediate artifacts.

### Prediction visualization

For one selected experiment, this analysis presents:

- Sample count and summary metrics.
- Predicted length versus measured length.
- A perfect-prediction reference line.
- Sorted predicted and measured series.
- Absolute and percentage error distributions.
- Error grouped by measured-length ranges.

### Error analysis

Error analysis ranks observations by percentage or absolute error. A selected failure can then be inspected alongside its image transformations and depth field.

This links model behavior back to possible causes such as poor detection, unusual pose, segmentation leakage, difficult background, rare fish type, or extreme length.

### Model comparison

Model comparison calculates the same metrics for every experiment under the same selected partition and optional fish-type filter. A leaderboard and metric chart make ablations visible.

### Fish-type comparison

For outdoor data, one experiment can be compared across fish types. This reveals whether a good overall score hides weak performance on a particular body form or underrepresented group.

### Model-by-type analysis

A two-dimensional comparison displays each model and fish-type combination. It is useful for finding methods that are consistently robust rather than strong only on the dominant types.

## 23. Intermediate Reuse and Lineage

Detection results, aligned images, depth fields, masks, isolated images, context, and image embeddings are expensive to produce. Retaining them allows later stages and repeated experiments to reuse prior work.

Reuse is scientifically valid only when the intermediate still corresponds to the same:

- Source image.
- Body-part interpretation.
- Model or extraction method.
- transformation policy.
- Feature meaning.

If an image, extraction method, or experimental setting changes, dependent intermediates should be regenerated. Otherwise a result may silently combine artifacts from different pipeline versions.

The split manifest should remain stable while recomputing image features. Changing both the split and preprocessing at once makes it impossible to attribute metric changes to one cause.

## 24. Rerun Principles

Use these principles when repeating work:

1. Preserve the source split for a continuing experiment series.
2. Derive all augmented observations from that persisted split.
3. Regenerate intermediates when their source image or extraction assumptions change.
4. Regenerate processed features after changing an upstream intermediate.
5. Retrain models after changing the processed population or feature meaning.
6. Compare predictions only when their split and target definitions match.
7. Record attrition and sample counts for every result.
8. Treat existing predictions as belonging to the exact data and preprocessing state that created them.

## 25. Scientific Limitations

### Monocular scale ambiguity

A single uncalibrated image cannot uniquely determine physical scale. Learned statistical relationships can estimate length for familiar capture conditions, but they do not eliminate the underlying ambiguity.

### Domain shift

Controlled and outdoor photographs have different scene distributions. Strong controlled performance does not guarantee outdoor performance.

New cameras, fishing practices, geographic regions, species distributions, or image-processing pipelines can create additional shifts.

### Dependence on upstream vision quality

Length prediction depends on detection, alignment, depth, and segmentation. A downstream model can be accurate on clean features while the complete system remains vulnerable to upstream errors.

The two trained detectors quantify part of this. Controlled detection is near-perfect, so controlled length error is essentially estimator error. Outdoor detection reaches roughly 85 percent precision and recall, so outdoor error is a compound of detector error and estimator error, and outdoor attrition is partly caused by the detector rather than by the photographs alone.

### Selection through attrition

Only successfully processed observations reach training. If difficult images fail more often, evaluation on complete cases may overstate performance on the full intended population.

### Augmentation is not new biological evidence

Zoom variants increase visual diversity but repeat the same fish and label. They must inherit the persisted split of their source and should not be counted as independent biological samples when interpreting effective sample size.

### Relative depth is not metric depth

Relative depth may improve learned predictions, but it does not provide physical distance without calibration.

### Fish-type imbalance

Types with many observations can dominate global results. Per-type metrics and sample counts are necessary, while separate models for rare types can be unstable.

### Context shortcuts

Scene context may correlate with fish length in the available data without representing a causal relationship. A model can learn capture habits, measuring surfaces, or type-specific backgrounds that do not generalize.

### Scene context is automatic annotation

Context values are produced by a vision-language model rather than observed. Their errors are silent, are not necessarily independent of scene type, and become part of the feature matrix. A gain from the rich outdoor bundle is a gain from geometry, mask shape, and automatic annotation together.

### Measurement quality

Ground-truth label error places a ceiling on predictive accuracy. Lengths were taken with a measuring tape by the individual photographers rather than under one supervised protocol, so tape placement, tail handling, and reading precision vary across contributors. Unit consistency and transcription quality are likewise part of the experimental system.

### Single split and single seed

Every reported number comes from one persisted split and one fixed random seed. That makes the comparison across feature families, depth conditions, and methods internally consistent, because all of them see the same training and test observations.

It also means no result carries a repeated-split variance estimate. A small difference between two nearby configurations can reflect split composition or initialization rather than a real advantage, and the risk grows with the number of compared columns, which is 17 for each controlled table and 41 for the outdoor table. Differences should be treated as informative when they are large, consistent across related conditions, and consistent with the ablation they belong to, and as inconclusive when they are small and isolated. Source-family bootstrap intervals quantify sampling uncertainty for the fixed split but do not replace repeated grouped partitions.

## 26. Recommended Reporting

A complete experiment report should include:

- Dataset family and data-collection setting.
- Number of source observations.
- Train, validation, and test counts.
- Number and percentage of observations surviving preprocessing.
- Fish-type distribution where applicable.
- Whether zoom-derived observations are included.
- Feature family and depth inclusion.
- Prediction method and global or type-specific scope.
- Detector performance for the dataset family, because it bounds feature quality.
- Test MAE, MAPE, RMSE, and R² with evaluated sample count and source-family bootstrap intervals.
- Per-type test metrics for outdoor data, marking groups whose predictions collapse to a constant.
- Comparison with the mean baseline.
- Examples of high-error and low-error observations.
- Known upstream failures and data limitations.

## 27. End-to-End Completion Checklist

An experiment series is complete when:

1. Source images and labels are present and consistent.
2. Unique image identities link every record to an image.
3. The source split is created and preserved.
4. Derived zoom families inherit their source split.
5. Required body parts are detected in the intended image coordinate system.
6. Alignment, depth, segmentation, isolated images, and context are available as required.
7. Engineered features have consistent meanings across all partitions.
8. Attrition is measured by split and fish type.
9. The full planned experiment matrix has produced predictions.
10. Test metrics are separated from training and validation diagnostics.
11. Global results are checked against fish-type and length-range behavior.
12. Representative errors are inspected against intermediate artifacts.
13. Results are interpreted within the limits of monocular scale estimation and the observed data distribution.

## 28. Glossary

**Ablation**  
A controlled comparison in which one information source, such as depth, is included in one experiment and excluded in another.

**Alignment**  
Rotation and cropping that place the tail-to-head direction horizontally.

**Attrition**  
Loss of observations between source data and the final processed population because required information could not be produced.

**Baseline**  
A simple reference predictor that more complex methods should outperform.

**Bounding box**  
A rectangle enclosing a detected image region.

**Complete case**  
An observation with every field required by a particular experiment.

**Convex hull**  
The smallest convex boundary enclosing a contour.

**Derived observation**  
A transformed version of a source image that keeps the same biological label and data lineage.

**Domain shift**  
A difference between the data distribution used for development and the distribution encountered later.

**Feature**  
A numerical or categorical measurement supplied to a prediction method.

**Fish-type-specific model**  
A predictor fitted only with observations from one fish type.

**Global model**  
A predictor fitted across all available fish types.

**Leakage**  
Use of information during training or selection that should be exclusive to held-out evaluation.

**Mask**  
A binary image indicating which pixels belong to the fish.

**Monocular depth**  
Depth inferred from one image using visual patterns rather than direct geometric measurement.

**Precision and recall**  
For a detector, the share of predicted landmarks that are correct and the share of true landmarks that are found.

**Scene annotation**  
A closed-set description of the original photograph produced automatically by a vision-language model rather than measured from pixels.

**Straight-line total length**  
The target measurement: the direct distance from the frontmost point of the head to the rearmost point of the tail, taken with a tape rather than along the body curve.

**Prediction table**  
The common result containing ground truth, split membership, and predictions from every experiment.

**Residual**  
Predicted length minus measured length.

**Solidity**  
Segmented contour area divided by the area of its convex hull.

**Split manifest**  
The record of which observations belong to training, validation, and test partitions.

**Test partition**  
Held-out observations used for final evaluation.

**Training partition**  
Observations used to estimate model parameters.

**Validation partition**  
Observations used to guide model selection without fitting the final reported test result.
