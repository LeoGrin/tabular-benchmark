def can_reach_end(nums):
 print(nums)
 if len(nums) == 1:
     return True
 if len(nums) > 1:
     if nums[0] == 0:
         return False
     else:
         return can_reach_end(nums[nums[0]:])

print(can_reach_end([1, 2, 3]))