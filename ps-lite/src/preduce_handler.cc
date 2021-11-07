#include "ps/server/preduce_handler.h"
#include <algorithm>
#include <unordered_map>

using namespace std::chrono;

namespace ps {

class PSHandler<PsfGroup::kPReduceScheduler>::ReduceStat {
public:
    ReduceStat(size_t max_workers, size_t ssp_bound, size_t sync_every) :
        max_workers_(max_workers), ssp_bound_(max_workers * ssp_bound),
        sync_every_(max_workers * sync_every) {
    }

    std::pair<std::vector<int>, bool> getPartner(int rank, float wait_time) {
        std::unique_lock<std::mutex> lock(mtx_);
        // must wait until the previous partial reduce decision finish
        while (critical_count) cv_.wait(lock);

        ready_workers_.push_back(rank);
        DisjointSetMerge(rank, ready_workers_[0]);
        if (need_global_sync_ || need_graph_sync_) {
            if (checkCondition()) {
                cv_.notify_all();
            } else {
                while (!checkCondition()) cv_.wait(lock);
            }
        } else {
            if (ready_workers_.size() == 1)
                // the first worker should set the wait time for others
                wake_time_ = system_clock::now() + microseconds(int(wait_time * 1000));
            if (ready_workers_.size() == max_workers_) {
                // if worker number is enough, notify all
                cv_.notify_all();
            } else {
                while (ready_workers_.size() < max_workers_ &&
                    cv_.wait_until(lock, wake_time_) == std::cv_status::no_timeout) {}
            }
        }
        // the first worker awake set the critical count
        if (!critical_count) {
            critical_count = ready_workers_.size();
            std::sort(ready_workers_.begin(), ready_workers_.end());
        }
        auto result = ready_workers_;
        auto is_sync = need_global_sync_;
        critical_count--;
        // if being the last thread, clear the state
        if (critical_count == 0) {
            if (need_graph_sync_) DisjointSetReset();
            size_t accum = accumulated_updates_ + ready_workers_.size();
            need_graph_sync_ = accum / ssp_bound_ > accumulated_updates_ / ssp_bound_;
            need_global_sync_ = accum / sync_every_ > accumulated_updates_ / sync_every_;
            accumulated_updates_ = accum;
            ready_workers_.clear();
            cv_.notify_all();
        }
        return std::make_pair(result, is_sync);
    }

    // register worker for initialization
    void registerWorker(int rank) {
        std::unique_lock<std::mutex> lock(mtx_);
        map_.emplace(rank, rank);
        if (map_.size() == max_workers_) {
            cv_.notify_all();
        } else {
            while (map_.size() < max_workers_) cv_.wait(lock);
        }
    }

    int critical_count = 0; // stop new worker from coming in, when the previous schedule is finishing
private:
    bool checkCondition() {
        if (need_global_sync_) {
            return ready_workers_.size() == max_workers_;
        }
        else if (need_graph_sync_) {
            return DisjointSetIsFullyMerged();
        }
        CHECK(false); // should not reach here
        return true;
    }
    //------------------------------- Implement disjoint set -------------------------------
    void DisjointSetReset() {
        for (auto& p : map_) {
            p.second = p.first;
        }
    }
    bool DisjointSetIsFullyMerged() {
        int head = DisjointSetFind(map_.begin()->first);
        for (const auto& p : map_) {
            if (DisjointSetFind(p.first) != head) return false;
        }
        return true;
    }
    int DisjointSetFind(int x) {
        return x == map_[x] ? x : (map_[x] = DisjointSetFind(map_[x]));
    }
    inline void DisjointSetMerge(int x, int y) {
        map_[DisjointSetFind(x)] = DisjointSetFind(y);
    }
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> ready_workers_;
    std::unordered_map<int, int> map_; // disjoint set
    system_clock::time_point wake_time_;
    const size_t max_workers_, ssp_bound_, sync_every_;
    size_t accumulated_updates_ = 0;
    bool need_graph_sync_ = false; // whether the next sync should connect all worker
    bool need_global_sync_ = false; // whether the next sync should be global
}; // store the state for every reduce key

PSHandler<PsfGroup::kPReduceScheduler>::~PSHandler() = default;

PSHandler<PsfGroup::kPReduceScheduler>::PSHandler() {}

void PSHandler<PsfGroup::kPReduceScheduler>::serve(
    const PSFData<kPReduceInit>::Request& request, PSFData<kPReduceInit>::Response& response) {
    Key k = get<0>(request);
    int rank = get<1>(request);
    size_t max_workers = get<2>(request);
    size_t ssp_bound = get<3>(request);
    size_t sync_every = get<4>(request);
    map_mtx_.lock();
    if (!map_.count(k))
        map_.emplace(k, std::unique_ptr<ReduceStat>(new ReduceStat(max_workers, ssp_bound, sync_every)));
    std::unique_ptr<ReduceStat>& obj = map_[k];
    map_mtx_.unlock();
    obj->registerWorker(rank);
    return;
}

void PSHandler<PsfGroup::kPReduceScheduler>::serve(
    const PSFData<kPReduceGetPartner>::Request& request,
    PSFData<kPReduceGetPartner>::Response& response) {
    Key k = get<0>(request);
    int rank = get<1>(request);
    float wait_time = get<2>(request);

    // get the reducestat
    map_mtx_.lock();
    CHECK(map_.count(k));
    std::unique_ptr<ReduceStat>& obj = map_[k];
    map_mtx_.unlock();

    auto result = obj->getPartner(rank, wait_time);
    // write return value
    get<0>(response).CopyFrom(result.first.data(), result.first.size());
    get<1>(response) = static_cast<int>(result.second);
    return;
}


} // namespace ps

