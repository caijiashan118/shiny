// AI城建系统前端JavaScript

let currentTab = 'basic-data';
let currentUploadType = '';

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 初始化应用
function initializeApp() {
    setupTabNavigation();
    loadProjectsData();
    loadFundsData();
    loadPivotData();
    setupCharts();
}

// 设置标签页导航
function setupTabNavigation() {
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

// 切换标签页
function switchTab(tabName) {
    // 隐藏所有标签页内容
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // 移除所有导航链接的active类
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // 显示选中的标签页
    document.getElementById(tabName).style.display = 'block';
    
    // 激活对应的导航链接
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    currentTab = tabName;
    
    // 根据标签页加载相应数据
    if (tabName === 'visualization') {
        loadVisualizationData();
    }
}

// 加载项目数据
async function loadProjectsData() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        displayProjectsData(data);
    } catch (error) {
        console.error('加载项目数据失败:', error);
        showAlert('加载项目数据失败', 'danger');
    }
}

// 显示项目数据
function displayProjectsData(data) {
    const tbody = document.querySelector('#projectsTable tbody');
    tbody.innerHTML = '';
    
    data.forEach(project => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${project.project_name}</td>
            <td>${project.project_code}</td>
            <td>${project.construction_unit}</td>
            <td>${project.project_department}</td>
            <td>${project.approval_time}</td>
            <td>${formatAmount(project.budget_amount)}</td>
            <td>${project.planned_start_time}</td>
            <td>${project.planned_completion_time}</td>
            <td><span class="status-badge status-${project.project_status}">${project.project_status}</span></td>
        `;
        tbody.appendChild(row);
    });
}

// 加载资金数据
async function loadFundsData() {
    try {
        const response = await fetch('/api/funds');
        const data = await response.json();
        displayFundsData(data);
    } catch (error) {
        console.error('加载资金数据失败:', error);
        showAlert('加载资金数据失败', 'danger');
    }
}

// 显示资金数据
function displayFundsData(data) {
    const tbody = document.querySelector('#fundsTable tbody');
    tbody.innerHTML = '';
    
    data.forEach(fund => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${fund.project_name}</td>
            <td>${fund.project_code}</td>
            <td>${fund.construction_unit}</td>
            <td>${fund.project_department}</td>
            <td>${formatAmount(fund.budget_amount)}</td>
            <td>${formatAmount(fund.fund_approval_amount)}</td>
            <td>${fund.approval_date}</td>
            <td><span class="fund-nature-badge fund-nature-${fund.fund_nature}">${fund.fund_nature}</span></td>
            <td>${fund.handler}</td>
            <td>${fund.handling_department}</td>
        `;
        tbody.appendChild(row);
    });
}

// 加载透视表数据
async function loadPivotData() {
    try {
        const response = await fetch('/api/pivot');
        const data = await response.json();
        displayPivotData(data);
    } catch (error) {
        console.error('加载透视表数据失败:', error);
        showAlert('加载透视表数据失败', 'danger');
    }
}

// 显示透视表数据
function displayPivotData(data) {
    const tbody = document.querySelector('#pivotTable tbody');
    tbody.innerHTML = '';
    
    data.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.project_code}</td>
            <td>${item.project_name}</td>
            <td>${formatAmount(item['一般债券'] || 0)}</td>
            <td>${formatAmount(item['专项债券'] || 0)}</td>
            <td><strong>${formatAmount(item['债券资金合计'] || 0)}</strong></td>
            <td>${formatAmount(item['中央综合财力'] || 0)}</td>
            <td>${formatAmount(item['土地出让金'] || 0)}</td>
            <td>${formatAmount(item['中央预算内投资'] || 0)}</td>
            <td>${formatAmount(item['中央补助'] || 0)}</td>
            <td>${formatAmount(item['省级补助'] || 0)}</td>
        `;
        tbody.appendChild(row);
    });
}

// 设置图表
function setupCharts() {
    // 这里可以初始化图表，暂时留空
}

// 加载可视化数据
async function loadVisualizationData() {
    try {
        const [projectsResponse, fundsResponse] = await Promise.all([
            fetch('/api/projects'),
            fetch('/api/funds')
        ]);
        
        const projectsData = await projectsResponse.json();
        const fundsData = await fundsResponse.json();
        
        createProjectStatusChart(projectsData);
        createFundNatureChart(fundsData);
        createInvestmentTrendChart(projectsData);
    } catch (error) {
        console.error('加载可视化数据失败:', error);
        showAlert('加载可视化数据失败', 'danger');
    }
}

// 创建项目状态图表
function createProjectStatusChart(data) {
    const ctx = document.getElementById('projectStatusChart').getContext('2d');
    
    // 统计项目状态
    const statusCount = {};
    data.forEach(project => {
        statusCount[project.project_status] = (statusCount[project.project_status] || 0) + 1;
    });
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(statusCount),
            datasets: [{
                data: Object.values(statusCount),
                backgroundColor: [
                    '#28a745', // 完工 - 绿色
                    '#ffc107', // 续建 - 黄色
                    '#17a2b8', // 在建 - 蓝色
                    '#dc3545'  // 暂停 - 红色
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// 创建资金性质图表
function createFundNatureChart(data) {
    const ctx = document.getElementById('fundNatureChart').getContext('2d');
    
    // 统计资金性质
    const fundCount = {};
    data.forEach(fund => {
        fundCount[fund.fund_nature] = (fundCount[fund.fund_nature] || 0) + 1;
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(fundCount),
            datasets: [{
                label: '资金安排数量',
                data: Object.values(fundCount),
                backgroundColor: [
                    '#007bff', '#28a745', '#ffc107', '#dc3545',
                    '#17a2b8', '#6f42c1', '#e83e8c'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// 创建投资趋势图表
function createInvestmentTrendChart(data) {
    const ctx = document.getElementById('investmentTrendChart').getContext('2d');
    
    // 按月份统计投资金额
    const monthlyInvestment = {};
    data.forEach(project => {
        const month = project.approval_time ? project.approval_time.substring(0, 7) : '未知';
        monthlyInvestment[month] = (monthlyInvestment[month] || 0) + (project.budget_amount || 0);
    });
    
    const sortedMonths = Object.keys(monthlyInvestment).sort();
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: sortedMonths,
            datasets: [{
                label: '投资金额(万元)',
                data: sortedMonths.map(month => monthlyInvestment[month] / 10000),
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// 导出项目数据
async function exportProjects() {
    try {
        const response = await fetch('/api/export/projects');
        const blob = await response.blob();
        downloadFile(blob, `项目基本信息_${new Date().toISOString().slice(0, 10)}.xlsx`);
    } catch (error) {
        console.error('导出项目数据失败:', error);
        showAlert('导出项目数据失败', 'danger');
    }
}

// 导出资金数据
async function exportFunds() {
    try {
        const response = await fetch('/api/export/funds');
        const blob = await response.blob();
        downloadFile(blob, `资金安排_${new Date().toISOString().slice(0, 10)}.xlsx`);
    } catch (error) {
        console.error('导出资金数据失败:', error);
        showAlert('导出资金数据失败', 'danger');
    }
}

// 导出透视表数据
async function exportPivot() {
    try {
        const response = await fetch('/api/export/pivot');
        const blob = await response.blob();
        downloadFile(blob, `资金安排透视表_${new Date().toISOString().slice(0, 10)}.xlsx`);
    } catch (error) {
        console.error('导出透视表数据失败:', error);
        showAlert('导出透视表数据失败', 'danger');
    }
}

// 显示上传模态框
function showUploadModal(type) {
    currentUploadType = type;
    const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
    modal.show();
}

// 上传文件
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('请选择文件', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const endpoint = currentUploadType === 'projects' ? '/api/upload/projects' : '/api/upload/funds';
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert(`上传成功，共导入 ${result.count} 条记录`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('uploadModal')).hide();
            
            // 重新加载数据
            if (currentUploadType === 'projects') {
                loadProjectsData();
            } else {
                loadFundsData();
            }
            loadPivotData();
        } else {
            showAlert(result.error || '上传失败', 'danger');
        }
    } catch (error) {
        console.error('上传文件失败:', error);
        showAlert('上传文件失败', 'danger');
    }
}

// 生成概览报表
async function generateOverviewReport() {
    try {
        const [projectsResponse, fundsResponse] = await Promise.all([
            fetch('/api/projects'),
            fetch('/api/funds')
        ]);
        
        const projectsData = await projectsResponse.json();
        const fundsData = await fundsResponse.json();
        
        const reportContent = generateOverviewReportContent(projectsData, fundsData);
        displayReport(reportContent);
    } catch (error) {
        console.error('生成概览报表失败:', error);
        showAlert('生成概览报表失败', 'danger');
    }
}

// 生成资金分析报表
async function generateFundReport() {
    try {
        const response = await fetch('/api/pivot');
        const data = await response.json();
        
        const reportContent = generateFundReportContent(data);
        displayReport(reportContent);
    } catch (error) {
        console.error('生成资金分析报表失败:', error);
        showAlert('生成资金分析报表失败', 'danger');
    }
}

// 生成趋势分析报表
async function generateTrendReport() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        
        const reportContent = generateTrendReportContent(data);
        displayReport(reportContent);
    } catch (error) {
        console.error('生成趋势分析报表失败:', error);
        showAlert('生成趋势分析报表失败', 'danger');
    }
}

// 生成概览报表内容
function generateOverviewReportContent(projectsData, fundsData) {
    const totalProjects = projectsData.length;
    const totalFunds = fundsData.length;
    const totalBudget = projectsData.reduce((sum, project) => sum + (project.budget_amount || 0), 0);
    const totalFundAmount = fundsData.reduce((sum, fund) => sum + (fund.fund_approval_amount || 0), 0);
    
    // 统计项目状态
    const statusCount = {};
    projectsData.forEach(project => {
        statusCount[project.project_status] = (statusCount[project.project_status] || 0) + 1;
    });
    
    return `
        <div class="report-section">
            <h4 class="report-title">项目概览报表</h4>
            <div class="row">
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${totalProjects}</div>
                        <div class="metric-label">项目总数</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${formatAmount(totalBudget)}</div>
                        <div class="metric-label">总概算金额(万元)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${totalFunds}</div>
                        <div class="metric-label">资金安排记录</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${formatAmount(totalFundAmount)}</div>
                        <div class="metric-label">总资金批复(万元)</div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-12">
                    <h5>项目状态分布</h5>
                    <div class="row">
                        ${Object.entries(statusCount).map(([status, count]) => `
                            <div class="col-md-3">
                                <div class="report-metric">
                                    <div class="metric-value">${count}</div>
                                    <div class="metric-label">${status}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 生成资金分析报表内容
function generateFundReportContent(data) {
    const totalProjects = data.length;
    const totalBondFunds = data.reduce((sum, item) => sum + (item['债券资金合计'] || 0), 0);
    const totalGeneralBonds = data.reduce((sum, item) => sum + (item['一般债券'] || 0), 0);
    const totalSpecialBonds = data.reduce((sum, item) => sum + (item['专项债券'] || 0), 0);
    
    return `
        <div class="report-section">
            <h4 class="report-title">资金分析报表</h4>
            <div class="row">
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${totalProjects}</div>
                        <div class="metric-label">涉及项目数</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${formatAmount(totalBondFunds)}</div>
                        <div class="metric-label">债券资金合计(万元)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${formatAmount(totalGeneralBonds)}</div>
                        <div class="metric-label">一般债券(万元)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="report-metric">
                        <div class="metric-value">${formatAmount(totalSpecialBonds)}</div>
                        <div class="metric-label">专项债券(万元)</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 生成趋势分析报表内容
function generateTrendReportContent(data) {
    // 按年份统计项目数量
    const yearlyProjects = {};
    data.forEach(project => {
        const year = project.approval_time ? project.approval_time.substring(0, 4) : '未知';
        yearlyProjects[year] = (yearlyProjects[year] || 0) + 1;
    });
    
    return `
        <div class="report-section">
            <h4 class="report-title">趋势分析报表</h4>
            <div class="row">
                <div class="col-12">
                    <h5>年度项目立项趋势</h5>
                    <div class="row">
                        ${Object.entries(yearlyProjects).map(([year, count]) => `
                            <div class="col-md-2">
                                <div class="report-metric">
                                    <div class="metric-value">${count}</div>
                                    <div class="metric-label">${year}年</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 显示报表
function displayReport(content) {
    const reportContent = document.getElementById('reportContent');
    const reportBody = document.getElementById('reportBody');
    
    reportBody.innerHTML = content;
    reportContent.style.display = 'block';
    
    // 滚动到报表区域
    reportContent.scrollIntoView({ behavior: 'smooth' });
}

// 格式化金额
function formatAmount(amount) {
    if (!amount) return '0';
    return (amount / 10000).toLocaleString('zh-CN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

// 下载文件
function downloadFile(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// 显示提示信息
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 3秒后自动消失
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 3000);
}