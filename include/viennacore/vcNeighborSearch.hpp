#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <vcKDTree.hpp>
#include <vcNFKDTree.hpp>
#include <vcVectorType.hpp>

namespace viennacore {

// Searches a snapshot of a 3D cloud using KDTree (default) or NFKDTree.
// NumericType: Coordinate type, typically float or double.
// Tree: KD-tree class template, e.g. NeighborSearch<double, NFKDTree>.
// Both backends return sorted (input index, Euclidean distance) pairs and
// accept an ordinary radius, including points on its boundary. The query point
// itself is included. Normals are validated but not used for searching.
template <class NumericType, template <class, class> class Tree = KDTree>
class NeighborSearch {
  using TreeType = Tree<NumericType, Vec3D<NumericType>>;
  static constexpr bool strictRadiusBoundary =
      std::is_same_v<TreeType, NFKDTree<NumericType, Vec3D<NumericType>>>;

public:
  using Neighbor = std::pair<std::size_t, double>;

  explicit NeighborSearch(const PointCloud<NumericType> &cloud) {
    cloud.validate();
    points_ = cloud.positions;
    tree_.setPoints(points_);
    tree_.build();
  }

  [[nodiscard]] std::vector<Neighbor> getKNN(std::size_t i, int k) const {
    if (i >= points_.size()) {
      VIENNACORE_LOG_ERROR("NeighborSearch::getKNN point index is invalid.");
    }
    if (k <= 0)
      return {};
    const auto neighbors = tree_.findKNearest(points_[i], k);
    return convertNeighbors(neighbors);
  }

  [[nodiscard]] std::vector<Neighbor> getRadius(std::size_t i,
                                                double radius) const {
    if (i >= points_.size()) {
      VIENNACORE_LOG_ERROR("NeighborSearch::getRadius point index is invalid.");
    }
    if (radius < 0 || std::isnan(radius))
      return {};
    auto searchRadius = static_cast<NumericType>(radius);
    searchRadius *= searchRadius;
    if constexpr (strictRadiusBoundary) {
      // nanoflann excludes the threshold itself; advance it to include points
      // at exactly the requested squared radius.
      searchRadius = std::nextafter(
          searchRadius, std::numeric_limits<NumericType>::infinity());
    }
    const auto neighbors =
        tree_.findNearestWithinRadius(points_[i], searchRadius);
    return convertNeighbors(neighbors);
  }

private:
  static std::vector<Neighbor> convertNeighbors(
      const std::optional<std::vector<std::pair<std::size_t, NumericType>>>
          &neighbors) {
    std::vector<Neighbor> result;
    if (!neighbors) {
      return result;
    }
    result.reserve(neighbors->size());
    for (const auto &[index, distance] : *neighbors) {
      const double convertedDistance = std::sqrt(static_cast<double>(distance));
      result.emplace_back(static_cast<std::size_t>(index), convertedDistance);
    }
    std::sort(result.begin(), result.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    return result;
  }

  std::vector<Vec3D<NumericType>> points_;
  TreeType tree_;
};

} // namespace viennacore
