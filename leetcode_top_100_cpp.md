### LeetCode Top 100 Cpp

##### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

```c++
class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		unordered_map<int, int> hash_map;
		vector<int> re;
		for (int i = 0; i < nums.size(); ++i) {
			int cp = target - nums[i];
			if (hash_map.find(cp) != hash_map.end()) {
				re.push_back(hash_map[cp]);
				re.push_back(i);
			}
			else {
				hash_map[nums[i]] = i;
			}
		}
        return re;

	}
};
```

##### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)


