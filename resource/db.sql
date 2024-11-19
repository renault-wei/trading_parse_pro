CREATE DATABASE `data_parse` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;

CREATE TABLE `data_parse`.`date_info` (
  `c_date` date NOT NULL,
  `years` int(11) DEFAULT NULL,
  `months` int(11) DEFAULT NULL,
  `days` int(11) DEFAULT NULL,
  `data_type` tinyint(1) DEFAULT NULL COMMENT '1 表示是周末，2 表示节假日，0 表示工作日',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`c_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- data_parse.holiday_info definition

CREATE TABLE `data_parse`.`holiday_info` (
  `holiday_id` int(11) NOT NULL AUTO_INCREMENT,
  `holiday_name` varchar(20) DEFAULT NULL COMMENT '假日名称',
  `holiday_time` date DEFAULT NULL COMMENT '假日时间',
  `remark` varchar(50) DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`holiday_id`)
) ENGINE=InnoDB AUTO_INCREMENT=117 DEFAULT CHARSET=utf8mb4 COMMENT='节假日';

-- data_parse.trade_table definition

CREATE TABLE `data_parse`.`trade_table` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `trade_date` date DEFAULT NULL,
  `trade_year` int(11) DEFAULT NULL,
  `trade_day` int(11) DEFAULT NULL,
  `trade_month` int(11) DEFAULT NULL,
  `trade_hour` int(11) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL,
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7321 DEFAULT CHARSET=utf8mb4;