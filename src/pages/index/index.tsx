import React, { useState, useEffect } from 'react';
import { View, Text, Button, ScrollView } from '@tarojs/components';
import Taro from '@tarojs/taro';
import './index.scss';

// ================= 1. 定义 TypeScript 接口 (契约) =================
// 药品的结构
interface Medication {
  id: string | number;
  name: string;
  dosage: string;
}

// 后端返回的单条冲突详情
interface ConflictDetail {
  drugA: string;
  drugB: string;
  severity: 'high' | 'medium' | 'low' | 'safe';
  description: string;
  suggestion: string;
}

// 后端返回的完整 AI 报告
interface AIReport {
  overallRiskLevel: 'high' | 'medium' | 'low' | 'safe';
  aiSummary: string;
  conflicts: ConflictDetail[];
}

// ================= 2. 主组件逻辑 =================
export default function MedicationList() {
  const [medications, setMedications] = useState<any[]>([]); // 你的 Medication 类型
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [reasoningSteps, setReasoningSteps] = useState<string[]>([]);
  const [report, setReport] = useState<any | null>(null);
  const [greeting, setGreeting] = useState('');

  useEffect(() => {
    // 动态问候语，增加产品温度
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('上午好');
    else if (hour < 18) setGreeting('下午好');
    else setGreeting('晚上好');

    const savedMeds = Taro.getStorageSync('my_medications');
    if (savedMeds) setMedications(savedMeds);
  }, []);

  // --- 调用摄像头扫码添加药物 ---
  const handleAddMedication = async () => {
    try {
      // 1. 调用微信原生扫码 API
      const res = await Taro.scanCode({
        scanType: ['barCode', 'qrCode'], // 允许扫描一维条形码和二维码
        onlyFromCamera: false, // 允许从相册选择（以防老人拍了照片让子女帮忙查）
      });

      // 拿到扫码结果（通常是药盒上的一串数字条码）
      const scannedBarcode = res.result;
      console.log('扫码成功，条码内容：', scannedBarcode);

      // 2. 交互提示：让老人知道系统正在努力工作
      Taro.showLoading({ title: '正在识别药盒...' });

      // 3. 模拟网络请求：将条码发送给后端，后端返回真实的药品信息
      // （实际开发中，这里你需要发一个 Taro.request 给后端）
      setTimeout(() => {
        Taro.hideLoading();

        // [模拟] 根据条码随机生成一个药名
        const mockDrugNames = ['阿司匹林肠溶片', '布洛芬缓释胶囊', '硝苯地平控释片', '二甲双胍肠溶片', '奥美拉唑肠溶胶囊'];
        // 用条码长度或最后一位随机取一个药名来模拟识别结果
        const randomName = mockDrugNames[scannedBarcode.length % mockDrugNames.length]; 

        // 组装新药物数据
        const newMed: Medication = { 
          id: Date.now(), 
          name: randomName, 
          dosage: '扫码默认剂量(请核对)' // 真实情况可能也能扫出规格，比如 100mg
        };

        // 更新状态和本地缓存
        const newList = [...medications, newMed];
        setMedications(newList);
        Taro.setStorageSync('my_medications', newList);
        
        // 扫码添加新药后，清空之前旧的检查报告
        setReport(null); 

        Taro.showToast({ title: '添加成功', icon: 'success', duration: 2000 });
      }, 1000); // 模拟 1 秒的网络延迟

    } catch (err) {
      // 如果用户中途点了左上角返回，取消了扫码，就会走到这里
      console.log('扫码取消或失败', err);
      // Taro.scanCode 取消时通常不抛出严重错误，所以我们可以静默处理，或者给个轻提示
      // Taro.showToast({ title: '已取消扫码', icon: 'none' });
    }
  };

  // --- 删除药物 ---
  const handleDelete = (id: string | number) => {
    const newList = medications.filter(med => med.id !== id);
    setMedications(newList);
    Taro.setStorageSync('my_medications', newList); // 同步更新缓存
    setReport(null); // 列表变动，清空旧的检查报告
  };

  // --- 核心：调用后端 AI 接口进行风险评估 ---
  const startCheck = () => {
    if (medications.length < 2) {
      Taro.showToast({ title: '至少需要两种药物才能检查冲突哦', icon: 'none' });
      return;
    }

    setIsLoading(true);
    setReport(null);
    setReasoningSteps(['正在初始化 AI 药师...', '正在读取您的用药清单...']);

    // TODO: 这里预留了真实的后端 API 请求位置
    const requestTask = Taro.request({
      url: 'https://你的后端域名/api/check-conflicts', // 替换为真实的后端接口
      method: 'POST',
      data: { medications: medications.map(m => m.name) },
      enableChunked: true, // 极其重要：开启流式接收（SSE适配）
      success: (res) => {
        // 请求彻底完成时的处理逻辑写在这里
        setIsLoading(false);
        // 注意：实际开发中，最终的结果可能会通过最后一块 chunk 发送过来
      },
      fail: (err) => {
        setIsLoading(false);
        Taro.showToast({ title: '网络连接失败，请重试', icon: 'error' });
      }
    });

    // 监听后端 LangGraph 传来的流式数据
    requestTask.onChunkReceived((response) => {
      // 这里的 response.data 是 ArrayBuffer，需要转码并解析
      // 预留解析逻辑：根据后端传来的结构，更新 reasoningSteps 或 finalReport
      console.log('接收到后端的数据块 (Chunk):', response.data);
      
      // 临时演示：模拟解析后的追加过程
      // setReasoningSteps(prev => [...prev, '解析到的最新动作...']);
    });
  };

  // ================= 3. 视图渲染 =================
  return (
    <View className='page-container'>
      {/* --- 全新的沉浸式头部设计 --- */}
      <View className='header-bg'>
        <View className='header-content'>
          <Text className='greeting'>{greeting}，</Text>
          <Text className='title'>这是您的用药清单</Text>
          <Text className='subtitle'>请核对药物，点击下方按钮由 AI 药师进行安全排查。</Text>
        </View>
      </View>

      <ScrollView scrollY className='main-content'>
        {/* --- 药物列表区 --- */}
        <View className='med-list-wrapper'>
          {medications.length === 0 ? (
            <View className='empty-state'>
              <Text className='empty-icon'>📦</Text>
              <Text className='empty-text'>您的药箱空空如也</Text>
              <Text className='empty-subtext'>点击下方添加按钮，扫一扫药盒条码</Text>
            </View>
          ) : (
            medications.map(med => (
              <View key={med.id} className='med-card'>
                <View className='med-icon-wrap'>💊</View>
                <View className='med-info'>
                  <Text className='med-name'>{med.name}</Text>
                  <Text className='med-desc'>每次 {med.dosage}</Text>
                </View>
                <View className='delete-btn' onClick={() => handleDelete(med.id)}>
                  <Text className='delete-text'>删除</Text>
                </View>
              </View>
            ))
          )}
        </View>

        {/* --- 优化后的添加按钮 --- */}
        <Button className='add-btn' onClick={handleAddMedication}>
          <Text className='add-icon'>+</Text> 扫码添加新药物
        </Button>

        {/* ... (保留你之前的 isLoading 和 report 的渲染逻辑) ... */}
        
      </ScrollView>

      {/* --- 底部悬浮操作区 --- */}
      <View className='bottom-bar'>
        <Button 
          className={`check-btn ${medications.length < 2 ? 'disabled' : ''}`} 
          onClick={startCheck}
          disabled={isLoading || medications.length < 2}
        >
          {isLoading ? 'AI 分析中...' : '开始安全检查'}
        </Button>
      </View>
    </View>
  );
}