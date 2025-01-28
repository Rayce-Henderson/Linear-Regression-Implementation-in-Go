package main

import (
"fmt"
"gonum.org/gonum/plot"
"gonum.org/gonum/math"
"gonum.org/gonum/机器学习/linear_regression"
)

func generateData(numSamples int, slope float64, intercept float64) ([]float64, []float64) {
randomSeed := 42
_, err := math.Seed(randomSeed)
if err != nil {
fmt.Printf("Error setting random seed: %v\n", err)
return
}

X := make([]float64, numSamples)
y := make([]float64, numSamples)

for i := 0; i < numSamples; i++ {
x := float64(i) / (numSamples - 1)
e := math.Random() * 0.2

X[i] = x
y[i] = intercept + slope*x + e
}

return X, y
}

func trainTestSplit(X []float64, y []float64, testSize float64) ([]float64, []float64, []float64, []float64) {
integerTestSize := int64(testSize * len(X))
if integerTestSize <= 0 || integerTestSize >= len(X) {
fmt.Printf("Error: test size must be between 0 and 1\n")
return
}

trainIndices := make([]int, len(X)-integerTestSize)
for i, _ := range trainIndices {
trainIndices[i] = i
}

random.Seed(42)
.shuffle(trainIndices)

var XTrain, XTest, yTrain, yTest []float64

for _, index := range trainIndices[:len(trainIndices)-(int64(integerTestSize)*1)] {
XTrain = append(XTrain, X[index])
yTrain = append(yTrain, y[index])
}

for i := len(trainIndices)-(int64(integerTestSize)*1); i < len(trainIndices); i++ {
XTest = append(XTest, X[i])
yTest = append(yTest, y[i])
}

return XTrain, XTest, yTrain, yTest
}

func calculateMSE(yPred, yActual []float64) float64 {
mse := 0.0
for i := range yPred {
mse += (yPred[i] - yActual[i]) * (yPred[i] - yActual[i])
}
return mse / float64(len(yPred))
}

func calculateR2(yActual, yPred []float64) float64 {
meanY := 0.0
for i := range yActual {
meanY += yActual[i]
}
meanY /= float64(len(yActual))

var numerator, denominator float64
for i := range yActual {
numerator += (yActual[i] - meanY) * (yPred[i] - meanY)
denominator += (yActual[i] - meanY) * (yActual[i] - meanY)
}

if denominator == 0 {
return 0.0
}

return numerator / denominator
}

func plotScatterAndRegression(X []float64, yActual []float64, yPred []float64) {
p := plot.New()
p.Title = "Linear Regression"
p.XLabel = "Feature"
p.YLabel = "Target"

points := plot.Data{}

for i := range X {
points.Add(i, yActual[i])
points.Add(i+0.1, yPred[i])
}

p.Add(points)

regLineX := make([]float64, 2)
regLineY := make([]float64, 2)
minX := math.Min(math.MinFloat64, X...)
maxX := math.Max(math.MaxFloat64, X...)

/regLineX[0] = minX - 0.1
/regLineX[1] = maxX + 0.1

// Calculate regression line values at minX and maxX
regLineY[0] = calculateRegression(regLineX[0])
regLineY[1] = calculateRegression(regLineX[1])

line := plot.NewLine(regLineX, regLineY)
line.LineStyle = plot.LineStyleDot
p.Add(line)

p.Draw()
}

func calculateRegression(x float64) float64 {
return model.Predict([]float64{x})
}

func main() {
// Generate synthetic data
X, y := generateData(100, 3, 2)

// Split into training and testing sets (80% train, 20% test)
XTrain, XTest, yTrain, yTest := trainTestSplit(X, y, 0.2)

// Train the linear regression model
model := machine_learning.LinearRegression.New(1) // 1 feature

if err := model.Fit(XTrain, yTrain); err != nil {
fmt.Printf("Error training model: %v\n", err)
return
}

// Make predictions on the test set
yPred := model.Predict(XTest)

// Calculate evaluation metrics
mse := calculateMSE(yPred, yTest)
r2 := calculateR2(yTest, yPred)

fmt.Printf("Mean Squared Error: %f\n", mse)
fmt.Printf("R² Score: %f\n", r2)

// Plot the results
.plotScatterAndRegression(XTest, yTest, yPred)
}
