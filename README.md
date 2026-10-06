                                             How this works ??
1. Model Pipeline

I recommend showing this visually as:

Video Input
↓
Face Detection — YuNet
↓
Face Features — ResNet18
↓
Voice Features — MFCC
↓
Face + Voice Models
↓
Late Fusion
↓
Stress / No Stress

And separately:

Facial Expression — FERPlus → Supporting verification

RAG → Grounded explanation

What each block means

1. Video Input
The user uploads a video containing a face and speech.

2. YuNet Face Detection
The system finds the person's face in selected video frames.

3. ResNet18 Face Encoder
The detected face is converted into numerical visual features.
Your ResNet18 produces a 512-dimensional feature vector.

4. MFCC Voice Features
The speech is extracted from the video and converted into audio features using:

MFCC
Delta
Delta-delta
Mean and standard deviation over time

This produces a 240-dimensional voice feature vector.

5. Separate Models
The two modalities are analyzed independently:

Face → MLP classifier
Voice → Logistic Regression classifier

Each produces a stress probability.

6. Late Fusion
The two probabilities are combined:

Final probability = 0.40 × Face + 0.60 × Voice

Then:

≥ 0.50 → Stress
< 0.50 → No Stress

This is your actual final decision mechanism.

2. FERPlus should NOT be shown as part of the stress pipeline

This is important for your project's technical accuracy.

FERPlus does facial-expression recognition, not stress prediction.

So UI should show it as an additional verification layer:

FERPlus Expression Verification

It looks at the detected face and estimates expressions such as:

Happy, Sad, Angry, Fear, Neutral, etc.

Then the UI can say something like:

Supporting evidence: The detected facial expression is consistent/inconsistent with the stress prediction.

But FERPlus does not change the final stress probability.

That separation makes your architecture look much more credible.

3. RAG should also be separate

RAG is not another ML classifier.

Its role is:

Prediction → Retrieve relevant project knowledge → Generate explanation

For example:

Prediction: Stress
Face probability: 0.71
Voice probability: 0.83
Expression: Fear
Explanation: The multimodal model predicts stress because both facial and vocal features show elevated stress probability. The detected fearful expression provides supporting evidence.

So the architecture is really:

                    ┌── Face → YuNet → ResNet18 → Face Classifier ──┐
Video ──────────────┤                                               │
                    └── Voice → MFCC → Voice Classifier ────────────┤
                                                                    ↓
                                                               Late Fusion
                                                                    ↓
                                                             Stress / No Stress
                                                                    │
                              ┌─────────────────────────────────────┤
                              ↓                                     ↓
                       FERPlus Verification                    RAG Explanation
                       Supporting evidence                    Grounded explanation
4. "How It Works" — simpler version for the website

This should not repeat the technical pipeline. It should explain the system to a visitor.

I would use these 5 steps:

01 — Capture

The system takes a video containing a person's face and speech.

02 — Analyze the Face

The face is detected and converted into visual features that the model can analyze.

03 — Analyze the Voice

The speech is converted into audio features that capture characteristics of the speaker's voice.

04 — Combine the Signals

The face and voice models independently estimate stress. Their results are combined to produce the final prediction.

05 — Verify & Explain

A facial-expression model provides additional supporting evidence, while the RAG layer provides a grounded explanation of the result.

5. What I recommend visually

Don't make it a huge paragraph.

Model Pipeline
Video
  ↓
YuNet
  ↓
ResNet18
  ↓
Face Model
  ↘
    Late Fusion → Stress / No Stress
  ↗
Voice Model
  ↑
MFCC
  ↑
Audio

Then below it:

How It Works

Capture → Analyze Face → Analyze Voice → Combine → Explain