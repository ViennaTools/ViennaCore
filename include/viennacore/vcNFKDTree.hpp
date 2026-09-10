#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanoflann.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vcLogger.hpp"
#include "vcVectorType.hpp"

namespace viennacore {

// 3D point data with a nanoflann dataset interface.
// NumericType: Floating-point coordinate type, typically float or double.
//
// NFKDTree copies only positions; normals and cached bounds are optional for
// that use. Call validate() explicitly when matching normals are required.
template <class NumericType> struct PointCloud {
  // Point coordinates, in the order used for returned neighbor indices.
  std::vector<Vec3D<NumericType>> positions;
  // Per-point normals; validate() requires one normal per position.
  std::vector<Vec3D<NumericType>> normals;
  // Optional bounds: {minimum corner, maximum corner}, enclosing all points.
  std::vector<Vec3D<NumericType>> min_max;

  // Returns the number of positions.
  [[nodiscard]] std::size_t size() const { return positions.size(); }

  // Checks normal count and optional bounding-box size.
  // Throws std::invalid_argument: If counts differ or nonempty min_max does not
  // contain exactly two corners. Does not check whether bounds enclose points.
  void validate() const {
    if (positions.size() != normals.size())
      throw std::invalid_argument(
          "PointCloud: positions and normals must have the same size.");
    if (!min_max.empty() && min_max.size() != 2)
      throw std::invalid_argument(
          "PointCloud: min_max must contain exactly two points.");
  }

  // nanoflann callback returning the number of indexed points.
  std::size_t kdtree_get_point_count() const { return size(); }
  // Returns a coordinate for nanoflann, without bounds checking.
  // idx: Position index, less than size().
  // dim: Coordinate axis, in [0, 3).
  NumericType kdtree_get_pt(std::size_t idx, std::size_t dim) const {
    return positions[idx][dim];
  }
  // Supplies cached bounds to nanoflann when available.
  // bb: Output bounding box with three low/high intervals.
  // Returns false if min_max is empty, requesting automatic computation.
  // Throws std::invalid_argument: If nonempty min_max has other than two
  // entries.
  template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const {
    if (min_max.empty())
      return false;
    if (min_max.size() != 2)
      throw std::invalid_argument(
          "PointCloud: min_max must contain exactly two points.");
    for (std::size_t i = 0; i < 3; ++i) {
      bb[i].low = min_max[0][i];
      bb[i].high = min_max[1][i];
    }
    return true;
  }
};

// Owning nanoflann KD-tree wrapper for nearest-neighbor searches.
// NumericType: Floating-point coordinate and squared-distance type.
// ValueType: Point container with size(), indexed coordinate access, and
// contiguous coordinates exposed by data(), e.g. std::vector or Vec3D.
// D: Compile-time dimension, or -1 to infer it from the input points.
// A positive D must match every stored point and query; this is not checked.
//
// Construction and setPoints() copy the input. Call build() before querying
// and after each setPoints(). Empty or unbuilt trees return std::nullopt.
// Neighbor pairs contain the original zero-based input index and the squared
// Euclidean distance. K-nearest results are sorted by distance; radius results
// are unsorted and exclude the radius boundary. These distance and radius
// conventions differ from KDTree, which returns ordinary Euclidean distances.
//
// Copy and move operations are disabled because the index references owned
// data. Concurrent const queries are supported after build(); do not modify or
// rebuild the tree while queries are running.
template <class NumericType, class ValueType = std::vector<NumericType>,
          int D = -1>
class NFKDTree {
public:
  using SizeType = typename std::vector<NumericType>::size_type;

private:
  struct Dataset {
    std::vector<ValueType> points;

    SizeType kdtree_get_point_count() const { return points.size(); }
    NumericType kdtree_get_pt(SizeType idx, SizeType dim) const {
      return points[idx][dim];
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  using Index = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<NumericType, Dataset>, Dataset, D, SizeType>;
  using Neighbor = std::pair<SizeType, NumericType>;

  Dataset data_;
  SizeType maxLeafSize_ = 10;
  std::unique_ptr<Index> tree_;

public:
  // Constructs an empty tree with a maximum leaf size of 10.
  NFKDTree() = default;

  // Copies points without building the search index.
  // points: Points of a common, nonzero dimension; may be empty.
  // maxLeafSize: Maximum points per leaf; zero is clamped to one.
  // Smaller leaves trade more tree traversal for fewer point comparisons.
  explicit NFKDTree(const std::vector<ValueType> &points,
                    SizeType maxLeafSize = 10)
      : maxLeafSize_(std::max(SizeType{1}, maxLeafSize)) {
    setPoints(points);
  }

  // Copies a cloud's positions without building the search index.
  // cloud: Source positions; normals and min_max are ignored, and
  // validate() is not called. Bounds are computed during build().
  // maxLeafSize: Maximum points per leaf; zero is clamped to one.
  explicit NFKDTree(const PointCloud<NumericType> &cloud,
                    SizeType maxLeafSize = 10)
      : maxLeafSize_(std::max(SizeType{1}, maxLeafSize)) {
    std::vector<ValueType> points;
    points.reserve(cloud.size());
    for (const auto &position : cloud.positions) {
      ValueType point{};
      if constexpr (std::is_same_v<ValueType, std::vector<NumericType>>)
        point.resize(3);
      if (point.size() != 3)
        throw std::invalid_argument("NFKDTree: PointCloud requires 3D points.");
      std::copy(position.begin(), position.end(), point.begin());
      points.push_back(std::move(point));
    }
    setPoints(points);
  }

  // The nanoflann index stores a reference to data_, so relocation is unsafe.
  NFKDTree(const NFKDTree &) = delete;
  NFKDTree &operator=(const NFKDTree &) = delete;
  NFKDTree(NFKDTree &&) = delete;
  NFKDTree &operator=(NFKDTree &&) = delete;

  // Replaces the owned points and invalidates the search index.
  // points: Points of a common, nonzero dimension matching D when fixed.
  // An empty vector clears the tree. Call build() to enable queries again.
  // Throws std::invalid_argument: For zero, oversized, or inconsistent point
  // dimensions. Validation failures leave the previous tree intact.
  void setPoints(const std::vector<ValueType> &points) {
    if (points.empty()) {
      tree_.reset();
      data_ = {};
      return;
    }
    const auto dimension = points.front().size();
    if (dimension == 0 ||
        dimension > static_cast<SizeType>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("NFKDTree: invalid point dimension.");
    for (const auto &point : points)
      if (point.size() != dimension)
        throw std::invalid_argument("NFKDTree: inconsistent point dimensions.");

    Dataset replacement{points};
    tree_.reset();
    data_ = std::move(replacement);
  }

  // Returns the owned point count, including before build().
  SizeType getNumberOfPoints() const { return data_.points.size(); }

  // Builds or rebuilds the index from the owned points.
  // Uses omp_get_max_threads() when OpenMP is enabled, otherwise one thread.
  // An empty tree logs a warning and remains unavailable for queries.
  void build() {
    tree_.reset();
    if (data_.points.empty()) {
      VIENNACORE_LOG_WARNING("NFKDTree: No points provided!");
      return;
    }
    unsigned int threads = 1;
#ifdef _OPENMP
    threads = static_cast<unsigned int>(omp_get_max_threads());
#endif
    auto index = std::make_unique<Index>(
        static_cast<int>(data_.points.front().size()), data_,
        nanoflann::KDTreeSingleIndexAdaptorParams(
            maxLeafSize_,
            nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
            threads));
    index->buildIndex();
    tree_ = std::move(index);
  }

  // Finds up to k nearest points, sorted by squared distance.
  // point: Query coordinates matching the stored dimension (unchecked).
  // k: Requested neighbor count;
  // Returns Pairs of input index and squared Euclidean distance, or
  // std::nullopt if the tree is unbuilt/empty or k <= 0. Ties have no
  // guaranteed index order.
  [[nodiscard]] std::optional<std::vector<Neighbor>>
  findKNearest(const ValueType &point, const int k) const {
    if (!tree_ || k <= 0)
      return {};
    std::vector<SizeType> indices(k);
    std::vector<NumericType> distances(k);
    const auto found =
        tree_->knnSearch(point.data(), k, indices.data(), distances.data());
    std::vector<Neighbor> result;
    result.reserve(found);
    for (SizeType i = 0; i < found; ++i)
      result.emplace_back(indices[i], distances[i]);
    return result;
  }

  // Finds the nearest point.
  // point: Query coordinates matching the stored dimension (unchecked).
  // Returns Input index and squared Euclidean distance, or std::nullopt for
  // an unbuilt/empty tree. Any equally near point may be returned in a tie.
  [[nodiscard]] std::optional<Neighbor>
  findNearest(const ValueType &point) const {
    const auto result = findKNearest(point, 1);
    if (!result || result->empty())
      return {};
    return result->front();
  }

  // Finds points strictly inside a squared-distance threshold.
  // point: Query coordinates matching the stored dimension (unchecked).
  // radiusSquared: Squared search radius: pass r * r for physical radius r.
  // Points exactly on the boundary are excluded; zero yields no matches.
  // expected: Allocation hint for the match buffer, not a result limit.
  // Returns Unsorted pairs of input index and squared Euclidean distance. An
  // engaged empty vector means no matches; std::nullopt means an unbuilt/empty
  // tree or a negative/NaN radiusSquared.
  [[nodiscard]] std::optional<std::vector<Neighbor>>
  findNearestWithinRadius(const ValueType &point, NumericType radiusSquared,
                          SizeType expected = 0) const {
    if (!tree_ || radiusSquared < 0 || std::isnan(radiusSquared))
      return {};

    std::vector<nanoflann::ResultItem<SizeType, NumericType>> matches;
    matches.reserve(expected);
    // NOTE: nanoflann uses a strict comparison; KDTree includes the radius
    // boundary.
    const auto found =
        tree_->radiusSearch(point.data(), radiusSquared, matches,
                            nanoflann::SearchParameters(0, false));
    std::vector<Neighbor> result;
    result.reserve(found);
    for (const auto &match : matches)
      result.emplace_back(match.first, match.second);
    return result;
  }
};

} // namespace viennacore
