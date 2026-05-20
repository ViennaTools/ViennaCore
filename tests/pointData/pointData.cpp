#include <vcPointData.hpp>
#include <vcTestAsserts.hpp>

using namespace viennacore;

int main() {
  PointData<double> pointData;

  PointData<double>::ScalarDataType scalars{1.0, 2.0, 3.0};
  pointData.insertNextScalarData(scalars, "TestScalars");
  VC_TEST_ASSERT(pointData.getScalarDataSize() == 1);
  VC_TEST_ASSERT(pointData.getScalarDataLabel(0) == "TestScalars");
  auto retrievedScalars = pointData.getScalarData("TestScalars");
  VC_TEST_ASSERT(retrievedScalars != nullptr);
  for (size_t i = 0; i < scalars.size(); ++i) {
    VC_TEST_ASSERT((*retrievedScalars)[i] == scalars[i]);
  }

  PointData<double>::VectorDataType vectors{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  pointData.insertNextVectorData(vectors, "TestVectors");
  VC_TEST_ASSERT(pointData.getVectorDataSize() == 1);
  VC_TEST_ASSERT(pointData.getVectorDataLabel(0) == "TestVectors");
  auto retrievedVectors = pointData.getVectorData("TestVectors");
  VC_TEST_ASSERT(retrievedVectors != nullptr);
  for (size_t i = 0; i < vectors.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      VC_TEST_ASSERT((*retrievedVectors)[i][j] == vectors[i][j]);
    }
  }

  PointData<double> mergeData;
  mergeData.insertNextScalarData({4.0, 5.0, 6.0}, "TestScalars");
  mergeData.insertNextVectorData({{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}},
                                 "TestVectors");
  pointData.mergeScalarData(mergeData);
  pointData.mergeVectorData(mergeData);

  retrievedScalars = pointData.getScalarData("TestScalars");
  VC_TEST_ASSERT(retrievedScalars != nullptr);
  for (size_t i = 0; i < scalars.size(); ++i) {
    VC_TEST_ASSERT((*retrievedScalars)[i] ==
                   scalars[i] + mergeData.getScalarData("TestScalars")->at(i));
  }

  retrievedVectors = pointData.getVectorData("TestVectors");
  VC_TEST_ASSERT(retrievedVectors != nullptr);
  for (size_t i = 0; i < vectors.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      VC_TEST_ASSERT((*retrievedVectors)[i][j] ==
                     vectors[i][j] +
                         mergeData.getVectorData("TestVectors")->at(i)[j]);
    }
  }

  PointData<double> appendData;
  appendData.insertNextScalarData({7.0, 8.0, 9.0}, "TestScalars");
  appendData.insertNextScalarData({7.0, 8.0, 9.0}, "TestScalars2");
  pointData.appendReplaceData(appendData);
  VC_TEST_ASSERT(pointData.getScalarDataSize() == 2);
  VC_TEST_ASSERT(pointData.getScalarDataLabel(0) == "TestScalars");
  VC_TEST_ASSERT(pointData.getScalarDataLabel(1) == "TestScalars2");
  retrievedScalars = pointData.getScalarData("TestScalars");
  VC_TEST_ASSERT(retrievedScalars != nullptr);
  for (size_t i = 0; i < scalars.size(); ++i) {
    VC_TEST_ASSERT((*retrievedScalars)[i] ==
                   appendData.getScalarData("TestScalars")->at(i));
  }
  retrievedScalars = pointData.getScalarData("TestScalars2");
  VC_TEST_ASSERT(retrievedScalars != nullptr);
  for (size_t i = 0; i < scalars.size(); ++i) {
    VC_TEST_ASSERT((*retrievedScalars)[i] ==
                   appendData.getScalarData("TestScalars2")->at(i));
  }
}