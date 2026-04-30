## First and second iterations of the agent.py loop - with kween2.5
Ran the model twice as a smoke test, no data yet, accuracy was 0 (expected).
# iter_0001_20260430_163329
Used a simple Convolutional Neural Network (CNN) with Batch Normalization and Dropout layers for regularization
optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
Used the correct directory but failed to load data.

# iter_0000_20260430_163003
Convolutional Neural Network (CNN) with several convolutional layers followed by fully connected layers.

optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=[metrics.Precision(name='precision'), 
                           metrics.Recall(name='recall')]

Used placeholder to simulate data loading and preprocessing steps, which will be replaced with actual data handling in future iterations.
