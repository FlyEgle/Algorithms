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
