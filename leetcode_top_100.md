### LeetCode Top 100
##### 1. 两数之和
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

####  160. 相交链表
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

#### 206. 反转链表
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
        lists[i]=lists[j]
        while i < j and lists[i] <=pivot:
            i += 1
        lists[j]=lists[i]
    lists[j] = pivot
    
    quick_sort(lists,low,i-1)
    quick_sort(lists,i+1,high)
    return lists
```

#### 15. 三数之和
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
```

#### 215. 数组中的第K个最大元素
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
```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```