import sys
import os
import json
import csv
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd


from APIs import *


class Node(object):
	def __init__(self, full_table, dict_in):
		'''
		construct tree
		'''

		self.full_table = full_table
		self.func = dict_in["func"]

		# row, num, str, obj, header, bool
		self.arg_type_list = APIs[self.func]["argument"]
		self.arg_list = []

		# [("text_node", a), ("func_node", b)]
		self.child_list = []
		child_list = dict_in["args"]

		assert len(self.arg_type_list) == len(child_list)

		# bool, num, str, row
		self.out_type = APIs[self.func]["output"]



		for each_child in child_list:
			if isinstance(each_child, str):
				self.child_list.append(("text_node", each_child))
			elif isinstance(each_child, dict):
				sub_func_node = Node(self.full_table, each_child)
				self.child_list.append(("func_node", sub_func_node))
			else:
				raise ValueError("child type error")


		self.result = None


	def eval(self):

		for each_child, each_type in zip(self.child_list, self.arg_type_list):
			if each_child[0] == "text_node":
				if each_child[1] == "all_rows":
					self.arg_list.append(self.full_table)
				else:
					self.arg_list.append(each_child[1])
			else:
				sub_result = each_child[1].eval()
				# print ("exit func: ", each_child[1].func)

				# invalid
				if isinstance(sub_result, ExeError):
					print ("sublevel error")
					return ExeError()
				elif each_type == "row":
					if not isinstance(sub_result, pd.DataFrame):
						print ("error function return type")
						return ExeError()
				elif each_type == "bool":
					if not isinstance(sub_result, bool):
						print ("error function return type")
						return ExeError()
				elif each_type == "str":
					if not isinstance(sub_result, str):
						print ("error function return type")
						return ExeError()


				self.arg_list.append(sub_result)


		result = APIs[self.func]["function"](*self.arg_list)
		return result



def execute_all(json_in):
	'''
	execute all logic forms
	'''

	with open(json_in) as f:
		data_in = json.load(f)

	num_all = 0
	num_correct = 0

	for data in tqdm(data_in):

		num_all += 1
		logic = data["logic"]

		table_header = data["table_header"]
		table_cont = data["table_cont"]

		try:
			pd_in = defaultdict(list)
			for ind, header in enumerate(table_header):
				for inr, row in enumerate(table_cont):

					# remove last summarization row
					if inr == len(table_cont) - 1 \
						and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
							"a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
						continue
					pd_in[header].append(row[ind])

			pd_table = pd.DataFrame(pd_in)
		except Exception:
			continue

		root = Node(pd_table, logic)
		res = root.eval()

		if res == True:
			num_correct += 1


	print ("All: ", num_all)
	print ("Correct: ", num_correct)

	print ("Correctness Rate: ", float(num_correct) / num_all)

	return num_all, num_correct





if __name__=='__main__':


	data_path = "../dataset/"
	all_data = data_path + "all_data.json"

	execute_all(all_data)









































