### LeetCode Top 100

##### 1. 两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dct = {}
        for i, n in enumerate(nums):
            cp = target - n 
            if cp in dct:
                return [dct[cp], i]
            else:
                dct[n] = i
```

##### 2. 两数相加
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
```python
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val 
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = p = ListNode(None)
        s = 0
      
        while l1 or l2 or s! =0:
            s += (l1.val if l1 else 0) + (l2.val if l2 else 0)
            p.next = ListNode(s%10)
            p= p.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

            s = s // 10
        return dummy.next
```

##### 3. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长连续子字符串 的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s:str) -> int:
        n = len(s)
        if n <= 1 :return n
        max_len, window = 0, set()
        left = right = 0
        while (right < n):
            if s[right] not in window:
                max_len = max(max_len, right-left + 1)
                window.add(s[right])
                right += 1
            else:
                window.remove(s[left])
                left += 1
        return max_len
```

##### 20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) == 1:
            return False 
        stack = [] # 栈
        for c in s:
#             print(stack)
            if c == "(" or c == "[" or c == "{":
                stack.append(c)
            elif c == ")":
                if "(" in stack:
                    if stack.pop() != "(":
                        return False
                else:
                    return False 
            elif c == "]":
                if "[" in stack:
                    if stack.pop() != "[":
                        return False
                else:
                    return False 
            elif c == "}":
                if "{" in stack:
                    if stack.pop() != "{":
                        return False
                else:
                    return False 
        if len(stack) == 0:
            return True
        else:
            return False 
```

#### 21. 合并两个有序链表
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dumpy = p = ListNode(0)
        p1, p2 = list1, list2 
        while p1 is not None and p2 is not None :
            if p1.val < p2.val :
                p.next = p1
                p1 = p1.next 
            else:
                p.next = p2
                p2 = p2.next 
          
            p = p.next

        if p1 is not None:
            p.next = p1
      
        if p2 is not None :
            p.next = p2

        return dumpy.next
```

#### 160. 相交链表
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a = headA 
        b = headB 

        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a 
```

#### 704. 二分查找

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while(l <= r):
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid 
            elif nums[mid] < target:
                l = mid +1
            elif nums[mid] > target:
                r = mid -1
        return -1
```

#### 69. x的平方根
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。
```python
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r = 0, x 
        ans = -1
        while (l <= r):
            m = l + (r - l )// 2
            s = m * m 
            if s <= x:
                l = m + 1
                ans = m
            else:
                r = m - 1
        return ans
```

#### 744. 寻找比目标字母大的最小字母
给你一个字符数组 letters，该数组按非递减顺序排序，以及一个字符 target。letters 里至少有两个不同的字符。

返回 letters 中大于 target 的最小的字符。如果不存在这样的字符，则返回 letters 的第一个字符。
```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        l, r = 0, len(letters)-1
      
        if target >= letters[-1]:
            return letters[0]
      
        while (l<r):
            m = l + (r - l) // 2
            if (letters[m] > target):
                r = m
            else:
                l = m + 1
              
        return letters[l]
```

#### 206. 反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None 
        cur = head 

        while cur:
            tmp = cur.next
            cur.next = pre 
            pre = cur 
            cur = tmp 
      
        return pre 
```

#### 141. 环形链表
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None or head.next is None:
            return False 
      
        slow = head 
        fast = head.next 
        while(slow != fast):
            if fast is None or fast.next is None:
                return False 
            fast = fast.next.next
            slow = slow.next
        return True 
```

#### 94. 二叉树中序遍历

```python
# 递归解法
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return [] 
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

# 迭代解法
class Solution:
    def inorderTraversal(self, root:TreeNode)->List[int]:
        res = []
        stack = []
        cur = root 
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left 
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
```

#### 斐波那契数列迭代法

```python
class Solution:
    def fib(self, n: int) -> int:
        MOD = 10 ** 9 + 7
        if n < 2:
            return n
        p, q, r = 0, 0, 1
        for i in range(2, n + 1):
            p = q
            q = r
            r = (p + q) % MOD
        return r
```

#### 剑指 Offer 22. 链表中倒数第k个节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        # 快慢指针
        slow, fast = head, head 
        while k:
            fast = fast.next 
            k-=1
      
        while fast:
            slow = slow.next 
            fast = fast.next 

        return slow
```

#### 88. 合并两个有序数组
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m +n - 1
        while (i>=0):
            if (m - 1) >=0 and (n-1) >= 0:
                if nums1[m-1] > nums2[n-1]:
                    nums1[i] = nums1[m-1]
                    m -= 1
                else:
                    nums1[i] = nums2[n-1]
                    n -= 1
            elif m-1 < 0:
                nums1[i] = nums2[n-1]
                n -= 1
            elif n - 1< 0:
                nums1[i] = nums1[m-1]
                m -=1 
            i-=1
```

#### 53.最大子数组和
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组
是数组中的一个连续部分。
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
      
        res = -inf 
        for i in range(len(nums)):
            res = max(res, dp[i])

        return res 
```

#### 226. 翻转二叉树
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        # 中序遍历，改left和right
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root 
```

#### 快速排序

```python
def partition(arr: List[int], low: int, high: int):
    pivot, j = arr[low], low
    for i in range(low + 1, high + 1):
        if arr[i] <= pivot:
            j += 1
            arr[j], arr[i] = arr[i], arr[j]
    arr[low], arr[j] = arr[j], arr[low]
    return j 

def quick_sort_between(arr: List[int], low: int, high: int):
    if high-low <= 1: # 递归结束条件
        return

    m = partition(arr, low, high)  # arr[m] 作为划分标准
    quick_sort_between(arr, low, m - 1)
    quick_sort_between(arr, m + 1, high)

def quick_sort(arr:List[int]):
    """
    快速排序(in-place)
    :param arr: 待排序的List
    :return: 快速排序是就地排序(in-place)
    """
    quick_sort_between(arr,0, len(arr) - 1)
```

```python
def quick_sort(lists,i,j):
    if i >= j:
        return lists
    pivot = lists[i]
    low = i
    high = j
    while i < j:
        while i < j and lists[j] >= pivot:
            j -= 1
        while i < j and lists[i] <=pivot:
            i += 1
        lists[j], lists[i] = lists[i], lists[j]
    
    lists[j] = pivot
    quick_sort(lists,low,i-1)
    quick_sort(lists,i+1,high)
    return lists
```

#### 15. 三数之和
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。
```python
class Solution:
    def threeSum(self, nums: [int]) -> [[int]]:
        nums.sort()
        res, k = [], 0
        for k in range(len(nums) - 2):
            if nums[k] > 0: break # 1. because of j > i > k.
            if k > 0 and nums[k] == nums[k - 1]: continue # 2. skip the same `nums[k]`.
            i, j = k + 1, len(nums) - 1
            while i < j: # 3. double pointer
                s = nums[k] + nums[i] + nums[j]
                if s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                elif s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
                else:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
        return res

```

#### 16. 最接近的三数之和
给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。

返回这三个数的和。

假定每组输入只存在恰好一个解。
```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        best = 10**7
      
        # 根据差值的绝对值来更新答案
        def update(cur):
            nonlocal best
            if abs(cur - target) < abs(best - target):
                best = cur
      
        # 枚举 a
        for i in range(n):
            # 保证和上一次枚举的元素不相等
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            # 使用双指针枚举 b 和 c
            j, k = i + 1, n - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                # 如果和为 target 直接返回答案
                if s == target:
                    return target
                update(s)
                if s > target:
                    # 如果和大于 target，移动 c 对应的指针
                    k0 = k - 1
                    # 移动到下一个不相等的元素
                    while j < k0 and nums[k0] == nums[k]:
                        k0 -= 1
                    k = k0
                else:
                    # 如果和小于 target，移动 b 对应的指针
                    j0 = j + 1
                    # 移动到下一个不相等的元素
                    while j0 < k and nums[j0] == nums[j]:
                        j0 += 1
                    j = j0

        return best
```

#### 二叉树层序遍历

##### BFS

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            size = len(queue)
            level = []
            for _ in range(size):
                cur = queue.popleft()
                if not cur:
                    continue
                level.append(cur.val)
                queue.append(cur.left)
                queue.append(cur.right)
            if level:
                res.append(level)
        return res

```

##### DFS

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        self.level(root, 0, res)
        return res

    def level(self, root, level, res):
        if not root: 
            return
        if len(res) == level: 
            res.append([])
      
        res[level].append(root.val)
        if root.left: 
            self.level(root.left, level + 1, res)
        if root.right: 
            self.level(root.right, level + 1, res)
```

#### Offer II 004. 只出现一次的数字

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        hash_map = dict()
        for num in nums:
            if num not in hash_map:
                hash_map[num] = 1
            else:
                hash_map[num] += 1
      
        for k, v in hash_map.items():
            if v == 1:
                return k
```

#### 14. 最长公共前缀
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        res = strs[0]
        i = 1
        while i < len(strs):
            while strs[i].find(res) != 0:
                res = res[0:len(res)-1]
            i += 1
        return res
```

#### 34. 在排序数组中查找元素的第一个和最后一个位置
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) < 1:
            return [-1, -1]

        if nums[0] > target or nums[-1] < target:
            return [-1, -1]
      
        # 二分
        left = 0
        right = len(nums) -1
        index = []
        while (left <= right):
            middle = (left + right) // 2
            if nums[middle] == target:
                left = middle 
                right = middle 
                while (left-1>=0 and nums[left-1] == target):
                    left -= 1
              
                while (right+1<=len(nums)-1 and nums[right+1] == target):
                    right += 1

                index = [left, right]
            elif nums[middle] > target:
                right = middle - 1 

            elif nums[middle] < target:
                left = middle + 1

        return index

class Solution(object):
    def searchRange(self,nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def left_func(nums,target):
            n = len(nums)-1
            left = 0
            right = n
            while(left<=right):
                mid = (left+right)//2
                if nums[mid] >= target:
                    right = mid-1
                if nums[mid] < target:
                    left = mid+1
            return left
        a =  left_func(nums,target)
        b = left_func(nums,target+1)
        if  a == len(nums) or nums[a] != target:
            return [-1,-1]
        else:
            return [a,b-1]
```

zzk: 这个地方我觉得还是拆成寻找左边界和右边界好点

参考： https://zhuanlan.zhihu.com/p/79553968

```python
class Solution:
    def searchRange(self, nums, target: int):
        left_border = self.getLeftBorder(nums, target)
        right_border = self.getRightBorder(nums, target)

        if left_border == len(nums) or nums[left_border] != target:
            return [-1, -1]
        else:
            return [left_border, right_border]

    def searchLeft(self, nums, target): 
        left = 0
        right = len(nums) # 注意这里我们right是len(nums)，所以我们搜索区间定为 [left, right)

        while left < right: 
            mid = (left + right) // 2
            """
            我们排出了 mid 这一项，因此分成两个区间去搜索
            [left, mid) [mid+1, right)
            """
            if nums[mid] == target: 
                # 继续在左半边搜索
                right = mid 
            if nums[mid] > target: 
                right = mid
            if nums[mid] < target: 
                left = mid + 1
        return left

    def searchRight(self, nums, target): 
        left = 0
        right = len(nums)

        while left < right: 
            mid = (left + right) // 2
            if nums[mid] == target: 
                # 继续在右半边搜索
                left = mid + 1
            if nums[mid] > target: 
                right = mid
            if nums[mid] < target: 
                left = mid + 1
        # 我们看第一个if判断，mid == target的时候，left = mid + 1。
        # 然后此时退出了while循环，left就越界了，而left-1则可能是target
        return left - 1

```


#### 215. 数组中的第K个最大元素
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        c = len(nums) - k
        def quick_sort(l, r):
            m = randint(l, r)
            nums[l], nums[m] = nums[m], nums[l]
            i, j = l, r
            while i < j:
                while i < j and nums[j] >= nums[l]: j -= 1
                while i < j and nums[i] <= nums[l]: i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l], nums[i] = nums[i], nums[l]

            if i > c: return quick_sort(l, i-1)
            elif i < c: return quick_sort(i+1, r)
            return nums[c]
        return quick_sort(0, len(nums)-1)
```

#### 234. 回文链表
给你一个单链表的头节点 head ，请你判断该链表是否为
回文链表
。如果是，返回 true ；否则，返回 false 。
```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # fallback 
        stack = []
        while head:
            stack.append(head.val)
            head = head.next 

        start, end = 0, len(stack) - 1
        while ( start <= end):
            if (stack[start] != stack[end]):
                return False 
            start += 1
            end -= 1
        return True
```

```python
class Solution:

    def find_link_mid(self, head):
        low, fast = head, head
        while low and low.next and fast and fast.next and fast.next.next:
            low, fast = low.next, fast.next.next
        return low

    def reverse_link(self, head):
        rev, rev_pre = head, None
        while rev:
            rev.next, rev_pre, rev =  rev_pre, rev, rev.next
        return rev_pre
      
    def isPalindrome(self, head: ListNode) -> bool:
        self.link_mid = self.find_link_mid(head)
        self.reverse_link = self.reverse_link(self.link_mid)
        ptr1, ptr2 = head, self.reverse_link
        while ptr1 and ptr2:
            if ptr1.val == ptr2.val:
                ptr1, ptr2 = ptr1.next, ptr2.next
            else:
                return False
        return True   
```

#### 338. 比特位计数
给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。
```python
# fallback
class Solution:
    def get2bins(self, num):
        out = []
        while num != 0:
            bins = num % 2
            num = num // 2
            out.append(bins)

        return sum(out)
    def countBits(self, n: int) -> List[int]:
        out_ = []
        for i in range(n+1):
            out_.append(self.get2bins(i))

        return out_
```

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        def countOnes(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1)
                ones += 1
            return ones
      
        bits = [countOnes(i) for i in range(n + 1)]
        return bits
```

#### 543. 二叉树的直径
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。
```python
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 1
        def depth(node):
            # 访问到空节点了，返回0
            if not node:
                return 0
            # 左儿子为根的子树的深度
            L = depth(node.left)
            # 右儿子为根的子树的深度
            R = depth(node.right)
            # 计算d_node即L+R+1 并更新ans
            self.ans = max(self.ans, L + R + 1)
            # 返回该节点为根的子树的深度
            return max(L, R) + 1

        depth(root)
        return self.ans - 1
```

#### 617. 合并二叉树
给你两棵二叉树： root1 和 root2 。

想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。

返回合并后的二叉树。

注意: 合并过程必须从两个树的根节点开始。
```python
class Solution(object):
    def mergeTrees(self, t1, t2):
        def dfs(r1, r2):
            if not (r1 and r2):
                return r1 if r1 else r2
            r1.val += r2.val 
            r1.left = dfs(r1.left, r2.left)
            r1.right = dfs(r1.right, r2.right)
            return r1

        return dfs(t1, t2)
```

#### 461. 汉明距离
两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。

给你两个整数 x 和 y，计算并返回它们之间的汉明距离。
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```

#### 121. 买卖股票的最佳时机
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
```python
# 动态规划
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0: return 0 # 边界条件
        dp = [0] * n
        minprice = prices[0] 

        for i in range(1, n):
            minprice = min(minprice, prices[i])
            dp[i] = max(dp[i - 1], prices[i] - minprice)

        return dp[-1]
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = float('inf')
        maxprofit = 0
        for price in prices:
            minprice = min(minprice, price)
            maxprofit = max(maxprofit, price - minprice)
        return maxprofit
```

#### 122. 买卖股票的最佳时机 II
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
  
        def isSorted(nums):
            size = len(nums)
            ans = 0
            for i in range(size - 1):
                if nums[i] < nums[i + 1]:
                    ans += 1 
            return ans 
      
        if not isSorted(prices):
            return 0 
      
        res = 0
      
        for i in range(1, len(prices)):
            res += max(0, prices[i] - prices[i-1])
      
        return res 
```

#### 101. 对称二叉树
给你一个二叉树的根节点 root ， 检查它是否轴对称。
```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        #左子树的左孩子==右子树的右孩子 and 左子树的右孩子 == 右子树的左孩子

        '''
        递归,自定义函数
        '''
        if root is None:
            return True
        return self.check(root.left,root.right)
  
    def check(self,left: TreeNode,right: TreeNode):
        #递归的终止条件是两个节点都为空
        #或左右有任意一个不为空，一定不对称
        #两个节点的值不相等
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        if left.val != right.val:
            return False
      
        return self.check(left.left,right.right) and self.check(left.right,right.left)
```

#### 169. 多数元素
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。
```python
# fallback
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        data = {}
        for n in nums:
            if n not in data:
                data[n] = 1
            else:
                data[n] += 1
          
        for k, v in data.items():
            if v > len(nums) / 2:
                return k 
```

#### 560. 和为K的子数组
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。
```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 要求的连续子数组
        count = 0
        n = len(nums)
        preSums = collections.defaultdict(int)
        preSums[0] = 1

        presum = 0
        for i in range(n):
            presum += nums[i]
          
            # if preSums[presum - k] != 0:
            count += preSums[presum - k]   # 利用defaultdict的特性，当presum-k不存在时，返回的是0。这样避免了判断

            preSums[presum] += 1  # 给前缀和为presum的个数加1
          
        return count
```

#### 279. 完全平方数
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
```python
# 背包问题，动态规划
class Solution:
    def numSquares(self, n: int) -> int:
        dp=[i for i in range(n+1)]
        for i in range(2,n+1):
            for j in range(1,int(i**(0.5))+1):
                dp[i]=min(dp[i],dp[i-j*j]+1)
        return dp[-1]
```

#### 322. 零钱兑换
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。
```python
# low performance
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = dict()
        def dp(n):
            if n == 0:return 0
            if n < 0 : return -1
            if n in memo:
                return memo[n]

            res = float("INF")
          
            for coin in coins:
                subproblem = dp(n - coin)
                if subproblem == -1:
                    continue
                res = min(res, subproblem+1)
          
            memo[n] = res if res != float('INF') else -1
            return memo[n]
      
        return dp(amount)
```

```python
# high performance
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount+1 for _ in range(amount+1)]
        dp[0] = 0

        for i in range(len(dp)):
            for c in coins:
                if i - c < 0:
                    continue
                dp[i] = min(dp[i], dp[i-c] + 1)

        if dp[amount] == amount + 1:
            return - 1
        else:
            return dp[amount]
```

#### 581. 最短无序连续子数组
给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

请你找出符合题意的 最短 子数组，并输出它的长度。
```python
# fallback
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)

        def isSorted() -> bool:
            for i in range(1, n):
                if nums[i - 1] > nums[i]:
                    return False
            return True
      
        if isSorted():
            return 0
      
        numsSorted = sorted(nums)
        left = 0
        while nums[left] == numsSorted[left]:
            left += 1

        right = n - 1
        while nums[right] == numsSorted[right]:
            right -= 1
      
        return right - left + 1
```

```python
# 一次遍历+min max计数
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        maxn, right = float("-inf"), -1
        minn, left = float("inf"), -1

        for i in range(n):
            if maxn > nums[i]:
                right = i
            else:
                maxn = nums[i]
          
            if minn < nums[n - i - 1]:
                left = n - i - 1
            else:
                minn = nums[n - i - 1]
      
        return 0 if right == -1 else right - left + 1
```

#### 647. 回文子串
给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。

回文字符串 是正着读和倒过来读一样的字符串。

子字符串 是字符串中的由连续字符组成的一个序列。
```python
# 动态规划
class Solution:
    def countSubstrings(self, s: str) -> int:
        # dp[i][j]: 以i开头，以j结尾的子串是否是回文串
        # if s[i] == s[j],
        #                   dp[i][j] = True if i == j       子串长度为1
        #                   dp[i][j] = True if j - i == 1   子串长度为2
        #                   dp[i][j] = dp[i+1][j-1]
        #  由于dp[i][j]依赖dp[i+1][j-1], dp矩阵要从左下角开始遍历
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = 0
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and ((j - i)< 2 || dp[i+1][j-1]):
                    dp[i][j] = True
                    ans += 1
        return ans


# 中心扩散
class Solution:
    def countSubstrings(self, s: str) -> int:
        # 以每个位置作为回文中心，尝试扩展
        # 回文中心有2种形式，1个数或2个数
        n = len(s)

        def spread(left, right):
            nonlocal ans
            while left >= 0 and right <= n - 1 and s[left] == s[right]:
                left -= 1
                right += 1
                ans += 1

        ans = 0
        for i in range(n):
            spread(i, i)
            spread(i, i + 1)

        return ans
```

##### 739. 每日温度
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
```python
# fallback 会超出时间限制
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        t = temperatures
        answer = [0 for _ in range(len(t))]
      
        for i in range(len(t) - 1):
            index = 0
            for j in range(i+1, len(t)):
                if t[j] > t[i]:
                    index = j - i 
                    break 
            answer[i] = index 
        answer[-1] = 0

        return answer

# 单调栈
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        length = len(temperatures)
        ans = [0] * length
        stack = []
        for i in range(length):
            temperature = temperatures[i]
            while stack and temperature > temperatures[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            stack.append(i)
        return ans
```

#### 5. 最长回文子串
给你一个字符串 s，找到 s 中最长的 
回文子串。
```python
# 动态规划
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
      
        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
      
        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break
                  
                if s[i] != s[j]:
                    dp[i][j] = False 
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
              
                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin:begin + max_len]

# 边界
class Solution:
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start: end + 1]
```

#### 11. 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans

```

#### 56. 合并区间
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
```

#### 霍夫变换

```python
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
from collections import Counter 

# 读取图像 
img = cv2.imread('line.png',2) 
img = 255 - img print(img.shape)  # [204,547] 

# 正变换：将xy坐标系中的点映射到极坐标中，记录映射函数经过的每一点 
def hough_forward_conver(x,y,points):   
	for t in range(0,360,2):       
		r = int(x * np.cos(np.pi*t/180) + y * np.sin(np.pi*t/180))       
		points.append([t,r])  # 直线经过的点放进去   
	return points   

# 反变换：根据极坐标系的坐标求xy坐标系的坐标 
def hough_reverse_conver(y, t,r):   
	x = int(- y * (np.sin(np.pi*t/180) / (np.cos(np.pi*t/180)+ 1e-4)) + r / (np.sin(np.pi*t/180)+1e-4))   
	return x   

# 霍夫正变换 
points = []  # 存放变换后的直线经过的点 
px, py = np.where(img == 255)  # 检测出直线上的点 
for x,y in zip(px,py):   
	points = hough_forward_conver(x,y,points)  # 霍夫变换，xy--->theta,rho 
print(len(points)) 

# 画极坐标图 
points = np.array(points) 
theta, rho = points[:,0], points[:,1] 
ax = plt.subplot(111, projection='polar') 
ax.scatter(np.pi*theta/180, rho, c='b', alpha=0.5,linewidths=0.01) 

# 霍夫空间的网格坐标系 
hough_space = np.zeros([360, 3000]) 
for point in points:
     t, r = point[0], point[1] + 1000  # r可能为负，防止索引溢出     h
     ough_space[t,r] += 1   

# 找出直线所在的点 
line_points = np.where(hough_space >= 15)
 print(len(line_points[0])) 

# 霍夫逆变换求xy 
mask = np.zeros_like(img) 
for t,r in zip(line_points[0],line_points[1]):
     for y in range(img.shape[0]):   
         x = hough_reverse_conver(y, t,r-1000)       
         if x in range(1,img.shape[1]):           
         	mask[y,x] += 1           

plt.imshow(mask) 
plt.imshow(img)
```

#### 198. 打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0

        size = len(nums)
        if size == 1:
            return nums[0]
      
        first, second = nums[0], max(nums[0], nums[1])
        for i in range(2, size):
            first, second = second, max(first + nums[i], second)
      
        return second
```

#### 128. 最长连续序列
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
```
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        hash = {key:key for key in nums}
        res = 0
        while hash:
            key, value = hash.popitem()
            up = key + 1
            down = key - 1
            length = 1
            while up in hash:
                length += 1
                del hash[up]
                up += 1
            while down in hash:
                length += 1
                del hash[down]
                down -= 1
            res = max(res, length)
        return res
```

#### 200. 岛屿数量
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    grid[r][c] = "0"
                    neighbors = collections.deque([(r, c)])
                    while neighbors:
                        row, col = neighbors.popleft()
                        for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                                neighbors.append((x, y))
                                grid[x][y] = "0"
      
        return num_islands

```

#### 98. 判断二叉搜索树是否合法

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(node, lower = float('-inf'), upper = float('inf')) -> bool:
            if not node:
                return True
          
            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True

        return helper(root)

```

#### 75. 颜色分类
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。
```python
# 单指针
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ptr = 0
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                nums[ptr], nums[i] = nums[i], nums[ptr]
                ptr+= 1
        for i in range(ptr, n):
            if nums[i] == 1:
                nums[ptr], nums[i] = nums[i], nums[ptr]
                ptr += 1
# 双指针
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ptr0, ptr1 = 0, 0 
        n = len(nums)

        for i in range(n):
            if nums[i] == 1:
                nums[ptr1], nums[i] = nums[i], nums[ptr1]
                ptr1 += 1
            elif nums[i] == 0:
                nums[ptr0], nums[i] = nums[i], nums[ptr0]
                if ptr0 < ptr1:
                    nums[ptr1], nums[i] = nums[i], nums[ptr1]
                ptr1 += 1
                ptr0 += 1
```

#### 139. 单词拆分
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。  
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        n = len(s)
        wordSet = set(wordDict)

        f = [False for _ in range(n+1)]
        f[0] = True 

        for i in range(n):
            for j in range(i+1, n+1):
                if f[i] and (s[i:j] in wordSet):
                    f[j] = True 
                    # break 
        return f[-1] 
```

#### 39. 组合总和
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 回殊法
        def dfs(candidates, begin, size, path, res, target):
            if target < 0:
                return
            if target == 0:
                res.append(path)
                return

            for index in range(begin, size):
                dfs(candidates, index, size, path + [candidates[index]], res, target - candidates[index])

        size = len(candidates)
        if size == 0:
            return []
        path = []
        res = []
        dfs(candidates, 0, size, path, res, target)

        return res 
```

#### 148. 排序链表
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。


```python
# 额外空间解法
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dumpy = p = ListNode(-1) # 头节点
        stack = []
        while head:
            stack.append(head.val)
            head = head.next 
      
        stack.sort()
        for n in stack:
            p.next = ListNode(n)
            p = p.next 

        return dumpy.next 

# 自顶向下， 采用分治+合并
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
            if not head:
                return head
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortFunc(head, mid), sortFunc(mid, tail))
          
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
      
        return sortFunc(head, None)

# 自底向上，归并
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
      
        if not head:
            return head
      
        length = 0
        node = head
        while node:
            length += 1
            node = node.next
      
        dummyHead = ListNode(0, head)
        subLength = 1
        while subLength < length:
            prev, curr = dummyHead, dummyHead.next
            while curr:
                head1 = curr
                for i in range(1, subLength):
                    if curr.next:
                        curr = curr.next
                    else:
                        break
                head2 = curr.next
                curr.next = None
                curr = head2
                for i in range(1, subLength):
                    if curr and curr.next:
                        curr = curr.next
                    else:
                        break
              
                succ = None
                if curr:
                    succ = curr.next
                    curr.next = None
              
                merged = merge(head1, head2)
                prev.next = merged
                while prev.next:
                    prev = prev.next
                curr = succ
            subLength <<= 1
      
        return dummyHead.next
```

#### 46. 全排列
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
```python
# 回溯 + DFS
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(size):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])

                    dfs(nums, size, depth + 1, path, used, res)

                    used[i] = False
                    path.pop()

        size = len(nums)
        if len(nums) == 0:
            return []

        used = [False for _ in range(size)]
        res = []
        dfs(nums, size, 0, [], used, res)
        return res

```

#### 114. 二叉树展开为链表
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        preorderList = list()

        def preorderTraversal(root: TreeNode):
            if root:
                preorderList.append(root)
                preorderTraversal(root.left)
                preorderTraversal(root.right)
      
        preorderTraversal(root)
        size = len(preorderList)
        for i in range(1, size):
            prev, curr = preorderList[i - 1], preorderList[i]
            prev.left = None
            prev.right = curr
```

#### 72. 编辑距离
给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
```python
# 动态规划+自底向上
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1+1)]
        for j in range(1, n2+1):
            dp[0][j] = dp[0][j-1] + 1
      
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i-1][0] + 1

        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
      
        return dp[-1][-1]
```

#### 239. 滑动窗口的最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)

        ans = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i - k:
                heapq.heappop(q)
            ans.append(-q[0][0])
        
        return ans


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
      
        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i -k:
                q.popleft()
            ans.append(nums[q[0]])

        return ans 
```

#### 42. 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
```python
# 双指针
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        ans = 0
        left_max, right_max = 0, 0
        while (left < right):
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if height[left] < height[right]:
                ans += left_max - height[left]
                left += 1
            else:
                ans += right_max - height[right]
                right -= 1

        return ans
```

#### 23. 合并K个升序链表
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。
```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return 
        n = len(lists)
        return self.merge(lists, 0, n-1)
  
    def merge(self, lists, left, right):
        if left == right:
            return lists[left]
        mid = left + (right - left) // 2
        l1 = self.merge(lists, left, mid)
        l2 = self.merge(lists, mid+1, right)
        return self.mergeTwoLists(l1, l2)
  
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dumpy = p = ListNode(0)
        p1, p2 = list1, list2 
        while p1 is not None and p2 is not None :
            if p1.val < p2.val :
                p.next = p1
                p1 = p1.next 
            else:
                p.next = p2
                p2 = p2.next 
          
            p = p.next

        if p1 is not None:
            p.next = p1
      
        if p2 is not None :
            p.next = p2

        return dumpy.next
```

#### 4. 寻找两个正序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 。


```python
# 暴力法
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 合并两个有序数组
        m = len(nums1)
        n = len(nums2)
        sort_nums = [0 for _ in range(n)]
        nums1 = nums1 + sort_nums
        self.merge(nums1, m, nums2, n)
        # print(nums1)
        if len(nums1) % 2 == 0:

            return (nums1[len(nums1) // 2 - 1] + nums1[len(nums1) // 2 ]) / 2
        else:
            print(int(len(nums1) / 2) + 1)
            return nums1[int(len(nums1) / 2)]
  

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m +n - 1
        while (i>=0):
            if (m - 1) >=0 and (n-1) >= 0:
                if nums1[m-1] > nums2[n-1]:
                    nums1[i] = nums1[m-1]
                    m -= 1
                else:
                    nums1[i] = nums2[n-1]
                    n -= 1
            elif m-1 < 0:
                nums1[i] = nums2[n-1]
                n -= 1
            elif n - 1< 0:
                nums1[i] = nums1[m-1]
                m -=1 
            i-=1
```

#### 105. 从前序与中序遍历序列构造二叉树
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

 
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def myBuildTree(preorder_left: int, preorder_right: int, inorder_left: int, inorder_right: int):
            if preorder_left > preorder_right:
                return None
          
            # 前序遍历中的第一个节点就是根节点
            preorder_root = preorder_left
            # 在中序遍历中定位根节点
            inorder_root = index[preorder[preorder_root]]
          
            # 先把根节点建立出来
            root = TreeNode(preorder[preorder_root])
            # 得到左子树中的节点数目
            size_left_subtree = inorder_root - inorder_left
            # 递归地构造左子树，并连接到根节点
            # 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
            root.left = myBuildTree(preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1)
            # 递归地构造右子树，并连接到根节点
            # 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
            root.right = myBuildTree(preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right)
            return root
      
        n = len(preorder)
        # 构造哈希映射，帮助我们快速定位根节点
        index = {element: i for i, element in enumerate(inorder)}
        return myBuildTree(0, n - 1, 0, n - 1)
```

#### 547. 省份数量
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。
```python
class DSU:
    def __init__(self, N):
        self.root = [i for i in range(N)]
      
    def find(self, k):
        if self.root[k] == k:
            return k
        return self.find(self.root[k])
  
    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        if x != y:
            self.root[y] = x
        return

class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        n = len(M)
        dsu = DSU(n)
        for i in range(n):
            for j in range(i+1, n):
                if M[i][j] == 1:
                    dsu.union(i, j)
        group = set()
        for i in range(n):
            group.add(dsu.find(i))
        return len(group)
```

#### 367. 有效的完全平方数
给你一个正整数 num 。如果 num 是一个完全平方数，则返回 true ，否则返回 false 。

完全平方数 是一个可以写成某个整数的平方的整数。换句话说，它可以写成某个整数和自身的乘积。

不能使用任何内置的库函数，如  sqrt 。
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left, right = 0, num
        while left <= right:
            mid = (left + right) // 2
            square = mid * mid
            if square < num:
                left = mid + 1
            elif square > num:
                right = mid - 1
            else:
                return True
        return False
```

#### 7. 整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。
```python
class Solution:
    def reverse(self, x: int) -> int:
        INT_MIN, INT_MAX = -2**31, 2**31 - 1

        rev = 0
        while x != 0:
            # INT_MIN 也是一个负数，不能写成 rev < INT_MIN // 10
            if rev < INT_MIN // 10 + 1 or rev > INT_MAX // 10:
                return 0
            digit = x % 10
            # Python3 的取模运算在 x 为负数时也会返回 [0, 9) 以内的结果，因此这里需要进行特殊判断
            if x < 0 and digit > 0:
                digit -= 10

            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能写成 x //= 10
            x = (x - digit) // 10
            rev = rev * 10 + digit
      
        return rev
```

#### 55. 跳跃游戏
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False

```

#### 134. 加油站

```python

# 贪心
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        spare = 0
        minSpare = 2**31
        minIndex = 0
      
        for i in range(n):
            spare += gas[i] - cost[i]
            if spare < minSpare:
                minSpare = spare
                minIndex = i 
      
        if spare < 0:
            return -1
        else:
            return (minIndex + 1) % n
```

#### 78. 子集

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
      
        res = [[]]
      
        for i in range(len(nums)):
            temp = []
            for r in res:
                temp.append(r + [nums[i]])
          
            res += temp 
          
        return res 
```

#### 61. 旋转链表

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k==0:
            return head 

        n = 1
        cur = head 
        while cur.next:
            cur = cur.next 
            n += 1
      
        if (n-k % n) == n:
            return head 

        add = n - k % n
        cur.next = head 

        while add:
            cur = cur.next 
            add -= 1

        ret = cur.next 
        cur.next = None 
        return ret 
```

#### 剑指 Offer 14- I. 剪绳子

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0 for _ in range(n+1)]
        dp[0], dp[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i] = max(dp[i], j * dp[i-j])

        ans = 1
        for i in range(1, n):
            ans = max(i * dp[n-i], ans)
      
        return ans 
```

#### 剑指 Offer 47. 礼物的最大价值

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:

        m = len(grid)
        n = len(grid[0])

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue 
                if i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += max(grid[i][j-1], grid[i-1][j])

        return grid[-1][-1]
```

#### 剑指 Offer 16. 数值的整数次方

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0: 
            return 0
        res = 1
        if n < 0: 
            x, n = 1 / x, -n
      
        while n:
            if n & 1: 
                res *= x
            x *= x
            n >>= 1

        return res
```

#### 面试题13. 机器人的运动范围

```python
class Solution:
    def dfs(i, j):
        if i >=m or j >= n:
            return 0
      
        if (i, j) in visited:
            return 0

        sum_i = i // 10 + i % 10
        sum_j = j // 10 + j % 10

        if sum_j + sum_i > k:
            return 0
      
        visited.add((i, j))
        return 1 + dfs(i+1, j) + dfs(i, j+1)
  

    visited = set()
    res = dfs(0, 0)
    return res 
```

#### [剑指 Offer 12. 矩阵中的路径](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def check(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
          
            visited.add((i, j))
            result = False
            for di, dj in directions:
                newi, newj = i + di, j + dj
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]):
                    if (newi, newj) not in visited:
                        if check(newi, newj, k + 1):
                            result = True
                            break
          
            visited.remove((i, j))
            return result

        h, w = len(board), len(board[0])
        visited = set()
        for i in range(h):
            for j in range(w):
                if check(i, j, 0):
                    return True
      
        return False
```

#### [剑指 Offer 60. n个骰子的点数](https://leetcode.cn/problems/nge-tou-zi-de-dian-shu-lcof/solution/jian-zhi-offer-60-n-ge-tou-zi-de-dian-sh-z36d/)

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [1 / 6] * 6
        for i in range(2, n + 1):
            tmp = [0] * (5 * i + 1)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j + k] += dp[j] / 6
            dp = tmp
        return dp
```

#### [面试题 01.05. 一次编辑](https://leetcode.cn/problems/one-away-lcci/)

```python
class Solution:
    def oneEditAway(self, first: str, second: str) -> bool:
        m, n = len(first), len(second)
        if m < n:
            return self.oneEditAway(second, first)
        if m - n > 1:
            return False
        for i, (x, y) in enumerate(zip(first, second)):
            if x != y:
                return first[i + 1:] == second[i + 1:] if m == n else first[i + 1:] == second[i:]  # 注：改用下标枚举可达到 O(1) 空间复杂度
        return True
```

#### [面试题 02.08. 环路检测](https://leetcode.cn/problems/linked-list-cycle-lcci/)

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow, fast = head, head
        while fast and fast.next: #开始走位
            slow = slow.next
            fast = fast.next.next
            if slow == fast: # 相遇
                break
          
        # 若无相会处，则无环路
        if not fast or not fast.next:
            return None
        # 若两者以相同的速度移动，则必然在环路起始处相遇
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
```

#### 26. [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

```python
class Solution:
    def removeDuplicates(self, nums) -> int:
        left = 0
        new_length = 1
        n = len(nums)
        for right in range(1, n):
            if nums[right] == nums[left]:
                continue
            else:
                left = left + 1
                nums[left] = nums[right]
                new_length += 1

        return new_length
```

#### 17. [电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    def letterCombinations(self, digits: str):
        if digits == "":
            return []

        table = {"2": ["a", "b", "c"],
                 "3": ["d", "e", "f"],
                 "4": ["g", "h", "i"],
                 "5": ["j", "k", "l"],
                 "6": ["m", "n", "o"],
                 "7": ["p", "q", "r", "s"],
                 "8": ["t", "u", "v"],
                 "9": ["w", "x", "y", "z"]}

        result = []
        combination = []
        num = len(digits)

        def backtrack(cnt):
            if cnt == num:
                result.append("".join(combination))
                return

            else:
                digit = digits[cnt]
                for char in table[digit]:
                    combination.append(char)
                    backtrack(cnt+1)
                    combination.pop()

        backtrack(0)
        return result
```

#### 22. [括号生成](https://leetcode.cn/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int):
        ans = []

        def backtrack(s, left, right):
            if len(s) == 2*n:
                ans.append("".join(s))

            if left < n:
                s.append("(")
                backtrack(s, left+1, right)
                s.pop()
            if right < left:
                s.append(")")
                backtrack(s, left, right+1)
                s.pop()

        backtrack([], 0, 0)
        return ans
```

#### 31. [下一个排列](https://leetcode.cn/problems/next-permutation/)

```python
class Solution:
    def nextPermutation(self, nums) -> None:
    
        i = len(nums) - 2

        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        left = i + 1
        right = len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

#### 628. [三个数的最大乘积](https://leetcode.cn/problems/maximum-product-of-three-numbers/)

```python
class Solution:
    def maximumProduct(self, nums) -> int:
        nums.sort()
        n = len(nums)
        return max(nums[0]*nums[1]*nums[n-1], nums[n-1]*nums[n-2]*nums[n-3])
```

#### 145. [二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return list()

        res = []
        stack = []
        prev = None
        while root or stack:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            if not root.right or root.right == prev:
                """
                两种情况，prev 标记上一次访问的是不是右子树
                1. 如果root没有右子树，则直接添加当前节点
                2. 如果右子树已经遍历过，则直接添加当前节点
                """
                res.append(root.val)
                prev = root
                root = None
            else:
                """
                说明右子树还没遍历完，因此再把root压回来
                """
                stack.append(root)
                root = root.right
        return res
```

#### 爬N阶楼梯，每次最多可以爬M阶，M<=N，问有多少种走法 （pdd）
```python
# 递归
def solution(n, m):
    way = 0
    if n == 0:
        return 1

    if (n >= m):
        for i in range(1, m+1):
            way += solution(n-i, m)

    else:
        way = solution(n, n)

    return way 
```

#### 146. [LRU缓存](https://leetcode.cn/problems/lru-cache/submissions/)
```python
class DListNode:
    def __init__(self, key, val, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next


class LRUCache:
    def __init__(self, capacity: int):
        """

        head -> tail
             <-
        After insert a Node:
        head -> Node1 -> tail
             <-       <-
        :param capacity:
        """
        self.hashmap = dict()
        self.capcity = capacity
        self.size = 0
        self.head = DListNode(0, 0)
        self.tail = DListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.hashmap:
            return -1
        node = self.hashmap[key]
        self.moveToHead(node)
        return node.val

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

    def put(self, key: int, value: int) -> None:
        if key not in self.hashmap:
            newNode = DListNode(key, value)
            self.hashmap[key] = newNode
            self.size += 1
            self.addToHead(newNode)

            if self.size > self.capcity:
                removedNode = self.removeTail()
                self.hashmap.pop(removedNode.key)
                self.size -= 1
        else:
            curNode = self.hashmap[key]
            curNode.val = value
            self.moveToHead(curNode)

```

#### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 左下角
        row, colw = len(matrix), len(matrix[0])
        
        i = row - 1
        j = 0
        while i >=0 and j <= colw - 1:
            if i == row - 1 and j == 0:
                # continue 
                i -= 1
                j += 1
            else:
                
                n = i
                for k in range(j):
                    matrix[i][k], matrix[row - k - 1][colw - i - 1] = matrix[row - k - 1][colw - i - 1], matrix[i][k]
                    
                i -= 1
                j += 1

        # 倒序
        matrix.reverse()
```

#### [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, columns = len(grid), len(grid[0])
        dp = [[0] * columns for _ in range(rows)]
        dp[0][0] = grid[0][0]
        for i in range(1, rows):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, columns):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, rows):
            for j in range(1, columns):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        
        return dp[rows - 1][columns - 1]
```